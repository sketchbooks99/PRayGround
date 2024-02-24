#include <prayground/prayground.h>

#include "params.h"

using namespace prayground;

// Utilities ------------------------------------------------------------------------------
extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Spectrum>;

static INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = getPayload<0>();
    const unsigned int u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

static INLINE DEVICE void traceSpectrum(
    OptixTraversableHandle handle, 
    const Vec3f& ro, const Vec3f& rd, 
    float tmin, float tmax, 
    unsigned int ray_type, 
    SurfaceInteraction* si, 
    float& lambda
)
{
    unsigned int u0, u1;
    unsigned int l = __float_as_int(lambda);
    packPointer(si, u0, u1);
    optixTrace(
        handle, 
        ro.toCUVec(), 
        rd.toCUVec(), 
        tmin, 
        tmax, 
        0.0f, 
        OptixVisibilityMask(1), 
        OPTIX_RAY_FLAG_NONE, 
        ray_type, 
        1, 
        ray_type, 
        u0, u1, 
        l
    );
}

// Raygen ------------------------------------------------------------------------------
static __forceinline__ __device__ float uniformSpectrumPDF()
{
    return 1.0f / (constants::max_lambda - constants::min_lambda);
}

// Raygen function
extern "C" __global__ void __raygen__spectrum()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int frame = params.frame;
    const Vec3ui idx(optixGetLaunchIndex());
    unsigned int seed = tea<4>(idx.x() * params.width + idx.y(), frame);

    float radiance;

    int i = params.samples_per_launch;

    // Uniform sampling of lambda
    float lambda = lerp(float(constants::min_lambda), float(constants::max_lambda), rnd(seed));

    do {
        const Vec2f subpixel_jitter = Vec2f(rnd(seed) - 0.5f, rnd(seed) - 0.5f);
        const Vec2f d = 2.0f * Vec2f(
            (static_cast<float>(idx.x()) + subpixel_jitter.x()) / static_cast<float>(params.width), 
            (static_cast<float>(idx.y()) + subpixel_jitter.y()) / static_cast<float>(params.height)
        ) - 1.0f;

        Vec3f ro, rd;
        getLensCameraRay(raygen->camera, d.x(), d.y(), ro, rd, seed);
        
        float throughput = 1.0f;

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = Spectrum{};
        si.albedo = Spectrum{};
        si.trace_terminate = false;

        int depth = 0;
        for (;;) {
            if (depth >= params.max_depth)
                break;

            traceSpectrum(params.handle, ro, rd, 0.01f, 1e16f, 0, &si, lambda);

            if (si.trace_terminate) {
                radiance += si.emission.getSpectrumFromWavelength(lambda) * throughput;
                break;
            }

            // Get emission from area emitter
            if (si.surface_info.type == SurfaceType::AreaEmitter) {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf,
                    &si, 
                    si.surface_info.data
                );
                radiance += si.emission.getSpectrumFromWavelength(lambda) * throughput;
                if (si.trace_terminate)
                    break;
            }
            // Sample scattering direction and evaluate bsdf
            else if (+(si.surface_info.type & SurfaceType::Material)) {
                float pdf;
                float bsdf = optixDirectCall<float, const float&, SurfaceInteraction*, void*, float&>(
                    si.surface_info.callable_id.bsdf, lambda, &si, si.surface_info.data, pdf);
                throughput *= bsdf / pdf;
            }

            ro = si.p;
            rd = si.wi;

            ++depth;
        }
    } while (--i);

    const uint32_t image_idx = idx.y() * params.width + idx.x();

    Vec3f xyz_result = Vec3f(
        radiance * CIE_X(lambda) / constants::CIE_Y_integral / uniformSpectrumPDF(),
        radiance * CIE_Y(lambda) / constants::CIE_Y_integral / uniformSpectrumPDF(),
        radiance * CIE_Z(lambda) / constants::CIE_Y_integral / uniformSpectrumPDF());

    if (!xyz_result.isValid())
        xyz_result = 0.0f;

    Vec3f color = XYZToSRGB(xyz_result);

    Vec3f accum_color = color / static_cast<float>(params.samples_per_launch);

    if (frame > 0) {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_color_prev = Vec3f(params.accum_buffer[image_idx]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_idx] = Vec4f(accum_color, 1.0f);
    Vec3u ucolor = make_color(accum_color);
    params.result_buffer[image_idx] = Vec4u(ucolor, 255);
}

// Miss ------------------------------------------------------------------------------
extern "C" __device__ void __miss__envmap()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    EnvironmentEmitter::Data* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
    SurfaceInteraction* si = getSurfaceInteraction();

    Ray ray = getWorldRay();
    const float lambda = __int_as_float(getPayload<2>());

    // Calculate intersection point at environment sphere
    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f*1e8f;
    const float D = half_b * half_b - a * c;
    const float sqrtD = sqrtf(D);
    const float t = (-half_b + sqrtD) / a;
    const Vec3f p = normalize(ray.at(t));

    // Get texture coordinates in sphere
    const float phi = atan2(p.z(), p.x());
    const float theta = asin(p.y());
    const float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    const float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    
    // Record hit-point information
    si->shading.uv = Vec2f(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    // Evaluate texture color for emittance
    si->emission = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data
    );
}

// Materials ------------------------------------------------------------------------------

/** 
 * @note Sellmeier equation of BK7 
 * @ref  https://www.thorlabs.co.jp/newgrouppage9.cfm?objectgroup_id=6973&tabname=N-BK7 
 **/
static __forceinline__ __device__ float bk7Index(const float& lambda)
{
    // Convert unit of wavelength: nm -> μm
    const float l = lambda * 0.001f;
    const float l2 = l * l;
    float ret = sqrtf(1.0f + ((1.03961212f * l2) / (l2 - 0.00600069867f)) + ((0.231792344f * l2) / (l2 - 0.0200179144f)) + ((1.01046945 * l2) / (l2 - 103.560653f)));
    if (ret != ret)
        ret = 1.5f;
    return ret;
}

static __forceinline__ __device__ float diamondIndex(const float& lambda)
{
    const float l2 = lambda * lambda;
    float ret = sqrtf(1.0f + ((0.3306f * l2) / (l2 - 175.0f * 175.0f)) + ((4.3346f * l2) / (l2 - 106.0f * 106.0f)));
    if (ret != ret)
        ret = 2.42f;
    return ret;
}

extern "C" __device__ float __direct_callable__sample_dielectric(const float& lambda, SurfaceInteraction* si, void* mat_data, float& pdf)
{
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(mat_data);

    // Sampling scattering direction
    float ni = 1.000292f;
    float nt = 1.5f;
    if (dielectric->sellmeier == Sellmeier::None)
        nt = dielectric->ior;
    else {
        if (dielectric->sellmeier == Sellmeier::Diamond)
            nt = diamondIndex(lambda);
        else if (dielectric->sellmeier == Sellmeier::BK7)
            nt = bk7Index(lambda);
    }
    float cosine = dot(si->wo, si->shading.n);
    bool into = cosine < 0;
    Vec3f outward_normal = into ? si->shading.n : -si->shading.n;

    if (!into) swap(ni, nt);

    cosine = fabs(cosine);
    float sine = sqrtf(1.0f - cosine * cosine);
    bool cannot_refract = ni * sine > nt;

    float reflect_prob = fresnel(cosine, ni, nt);
    uint32_t seed = si->seed;

    // Sampling refracting/reflecting direction depends on fresnel reflectance
    if (cannot_refract || reflect_prob > UniformSampler::get1D(seed))
        si->wi = reflect(si->wo, outward_normal);
    else
        si->wi = refract(si->wo, outward_normal, cosine, ni, nt);
    si->trace_terminate = false;
    si->seed = seed;

    // PDF evaluation 
    pdf = 1.0f;

    // BSDF evaluation
    si->emission = Spectrum{};
    Spectrum albedo = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(dielectric->texture.prg_id, si, dielectric->texture.data);
    return albedo.getSpectrumFromWavelength(lambda);
}

// Diffuse 
extern "C" __device__ float __direct_callable__sample_diffuse(const float& lambda, SurfaceInteraction* si, void* mat_data, float& pdf)
{
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(mat_data);

    if (diffuse->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);
    
    uint32_t seed = si->seed;
    const float z0 = rnd(seed);
    const float z1 = rnd(seed);
    Vec3f wi = cosineSampleHemisphere(z0, z1);
    Onb onb(si->shading.n);
    onb.inverseTransform(wi);
    si->wi = normalize(wi);
    si->seed = seed;
    si->trace_terminate = false;
    
    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    // PDF evaluation
    pdf = cosine * math::inv_pi;

    // BSDF evaluation
    const Spectrum albedo = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(diffuse->texture.prg_id, si, diffuse->texture.data);
    si->emission = Spectrum{};
    return albedo.getSpectrumFromWavelength(lambda) * pdf;
}

// Disney
static __forceinline__ __device__ float disneyBRDF(
    const Disney::Data* disney, const Spectrum& base_spectrum, const float base,  
    const Vec3f& V, const Vec3f& L, const Vec3f& N, const Vec3f& H, 
    const float NdotV, const float NdotL, const float NdotH, const float LdotH, 
    const Vec3f& X, const Vec3f& Y)
{
    if (NdotV <= 0.0f || NdotL <= 0.0f)
        return 0.0f;

    // Diffuse term (diffuse, subsurface, sheen) ======================
    // Diffuse
    const float Fd90 = 0.5f + 2.0f * disney->roughness * LdotH*LdotH;
    const float FVd90 = fresnelSchlickT(NdotV, Fd90);
    const float FLd90 = fresnelSchlickT(NdotL, Fd90);
    const float f_diffuse = base * math::inv_pi * FVd90 * FLd90;

    // Subsurface
    const float Fss90 = disney->roughness * LdotH*LdotH;
    const float FVss90 = fresnelSchlickT(NdotV, Fss90);
    const float FLss90 = fresnelSchlickT(NdotL, Fss90); 
    const float f_subsurface = base * math::inv_pi * 1.25f * (FVss90 * FLss90 * ((1.0f / (NdotV * NdotL)) - 0.5f) + 0.5f);

    // Sheen
    const float lumi = base_spectrum.y() / constants::CIE_Y_integral;
    const float rho_tint = base / lumi;
    const float rho_sheen = lerp(1.0f, rho_tint, disney->sheen_tint);
    const float f_sheen = disney->sheen * rho_sheen * pow5(1.0f - LdotH);

    // Specular term (specular, clearcoat) ============================
    // Spcular
    const float alpha = fmaxf(0.001f, disney->roughness);
    const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
    const float ax = fmaxf(0.001f, pow2(alpha) / aspect);
    const float ay = fmaxf(0.001f, pow2(alpha) * aspect);
    const float rho_specular = lerp(1.0f, rho_tint, disney->specular_tint);
    const float Fs0 = lerp(0.08f * disney->specular * rho_specular, base, disney->metallic);
    const float FHs0 = fresnelSchlickR(LdotH, Fs0);
    const float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);
    const float f_specular = FHs0 * Ds * Gs;

    // Clearcoat
    const float Fcc = fresnelSchlickR(LdotH, 0.04f);
    const float alpha_cc = lerp(0.001f, 0.1f, disney->clearcoat_gloss);
    const float Dcc = GTR1(NdotH, alpha_cc);
    const float Gcc = smithG_GGX(NdotV, 0.25f);
    const float f_clearcoat = 0.25f * disney->clearcoat * Fcc * Dcc * Gcc;

    const float out = ( 1.0f - disney->metallic ) * ( lerp( f_diffuse, f_subsurface, disney->subsurface ) + f_sheen ) + f_specular + f_clearcoat;
    return out * clamp(NdotL, 0.0f, 1.0f);
}

static __forceinline__ __device__ float disneyPDF(const Disney::Data* disney, const float NdotH, const float NdotL)
{
    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    const float specular_ratio = 1.0f - diffuse_ratio;

    const float alpha = fmaxf(0.001f, disney->roughness);
    const float alpha_cc = lerp(0.001f, 0.1f, disney->clearcoat_gloss);

    const float pdf_Ds = GTR2(NdotH, alpha);
    const float pdf_Dcc = GTR1(NdotH, alpha_cc);
    const float ratio = 1.0f / (1.0f + disney->clearcoat);
    const float pdf_specular = lerp(pdf_Dcc, pdf_Ds, ratio);
    const float pdf_diffuse = NdotL * math::inv_pi;

    return diffuse_ratio * pdf_diffuse + specular_ratio * pdf_specular;
}

extern "C" __device__ float __direct_callable__sample_disney(const float& lambda, SurfaceInteraction* si, void* mat_data, float& pdf)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);
    if (disney->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    unsigned int seed = si->seed;
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    Onb onb(si->shading.n);

    if (rnd(seed) < diffuse_ratio)
    {
        Vec3f wi = cosineSampleHemisphere(z1, z2);
        onb.inverseTransform(wi);
        si->wi = normalize(wi);
    }
    else
    {
        float gtr2_ratio = 1.0f / (1.0f + disney->clearcoat);
        Vec3f h;
        const float alpha = fmaxf(0.001f, disney->roughness);
        if (rnd(seed) < gtr2_ratio)
            h = sampleGGX(z1, z2, alpha);
        else
            h = sampleGTR1(z1, z2, alpha);
        onb.inverseTransform(h);
        si->wi = normalize(reflect(si->wo, h));
    }
    si->trace_terminate = false;
    si->seed = seed;

    const Vec3f V = -normalize(si->wo);
    const Vec3f L = normalize(si->wi);
    const Vec3f N = normalize(si->shading.n);
    const Vec3f H = normalize(V + L);
    const float NdotV = fabs(dot(N, V));
    const float NdotL = fabs(dot(N, L));
    const float NdotH = dot(N, H);
    const float LdotH /* = VdotH */ = dot(L, H);

    // PDF evaluation
    pdf = disneyPDF(disney, NdotH, NdotL);

    // BSDF evaluation
    const Spectrum base_spectrum = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(
        disney->base.prg_id, si, disney->base.data);
    const float base = base_spectrum.getSpectrumFromWavelength(lambda);
    return disneyBRDF(disney, base_spectrum, base, 
        V, L, N, H, NdotV, NdotL, NdotH, LdotH, 
        si->shading.dpdu, si->shading.dpdv);
}

// Area emitter
extern "C" __device__ void __direct_callable__area_emitter(SurfaceInteraction* si, void* surface_data)
{
    const AreaEmitter::Data* area = reinterpret_cast<AreaEmitter::Data*>(surface_data);
    si->trace_terminate = true;
    float is_emitted = 1.0f;
    if (!area->twosided)
        is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    
    const Spectrum base = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(
        area->texture.prg_id, si, area->texture.data);
    si->albedo = base;
    si->emission = base * area->intensity * is_emitted;
}

// Texture functions ---------------------------------------------------------------
extern "C" __device__ Spectrum __direct_callable__constant(SurfaceInteraction* si, void* tex_data)
{
    return pgGetConstantTextureValue<SampledSpectrum>(si->shading.uv, tex_data);
}

extern "C" __device__ Spectrum __direct_callable__checker(SurfaceInteraction* si, void* tex_data)
{
    return pgGetCheckerTextureValue<SampledSpectrum>(si->shading.uv, tex_data);
}

extern "C" __device__ Spectrum __direct_callable__bitmap(SurfaceInteraction* si, void* tex_data)
{
    return pgGetBitmapTextureValue<SampledSpectrum>(si->shading.uv, tex_data);
}

// Hitgroup functions ---------------------------------------------------------------
extern "C" __device__ void __closesthit__mesh()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const TriangleMesh::Data* mesh = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    const int id = optixGetPrimitiveIndex();
    const Face face = mesh->faces[id];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const Vec3f p0 = mesh->vertices[face.vertex_id.x()];
    const Vec3f p1 = mesh->vertices[face.vertex_id.y()];
    const Vec3f p2 = mesh->vertices[face.vertex_id.z()];

    const Vec2f texcoord0 = mesh->texcoords[face.texcoord_id.x()];
    const Vec2f texcoord1 = mesh->texcoords[face.texcoord_id.y()];
    const Vec2f texcoord2 = mesh->texcoords[face.texcoord_id.z()];
    const Vec2f texcoords = (1 - u - v) * texcoord0 + u * texcoord1 + v * texcoord2;

    const Vec3f n0 = mesh->normals[face.normal_id.x()];
    const Vec3f n1 = mesh->normals[face.normal_id.y()];
    const Vec3f n2 = mesh->normals[face.normal_id.z()];

    const Vec3f local_n = (1 - u - v) * n0 + u * n1 + v * n2;
    const Vec3f world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n.toCUVec()));

    auto si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = texcoords;
    si->surface_info = data->surface_info;

    Vec3f dpdu, dpdv;
    const Vec2f duv02 = texcoord0 - texcoord2;
    const Vec2f duv12 = texcoord1 - texcoord2;
    const Vec3f dp02 = p0 - p2;
    const Vec3f dp12 = p1 - p2;
    const float D = duv02.x() * duv12.y() - duv02.y() * duv12.x();
    bool degenerateUV = abs(D) < 1e-8f;
    if (!degenerateUV)
    {
        const float invD = 1.0f / D;
        dpdu = (duv12.y() * dp02 - duv02.y() * dp12) * invD;
        dpdv = (-duv12.x() * dp02 + duv02.x() * dp12) * invD;
    }
    if (degenerateUV || length(cross(dpdu, dpdv)) == 0.0f)
    {
        const Vec3f n = normalize(cross(p2 - p0, p1 - p0));
        Onb onb(n);
        dpdu = onb.tangent;
        dpdv = onb.bitangent;
    }
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu.toCUVec()));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv.toCUVec()));
}

extern "C" __device__ void __intersection__sphere()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

    const Vec3f center = sphere->center;
    const float radius = sphere->radius;
    
    Ray ray = getLocalRay();

    const Vec3f oc = ray.o - center;
    const float a = dot(ray.d, ray.d);
    const float half_b = dot(oc, ray.d);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant > 0.0f) {
        float sqrtd = sqrtf(discriminant);
        float t1 = (-half_b - sqrtd) / a;
        bool check_second = true;
        if (t1 > ray.tmin && t1 < ray.tmax) {
            Vec3f normal = normalize((ray.at(t1) - center) / radius);
            check_second = false;
            optixReportIntersection(t1, 0, Vec3f_as_ints(normal));
        }

        if (check_second) {
            float t2 = (-half_b + sqrtd) / a;
            if (t2 > ray.tmin && t2 < ray.tmax) {
                Vec3f normal = normalize((ray.at(t2) - center) / radius);
                optixReportIntersection(t2, 0, Vec3f_as_ints(normal));
            }
        }
    }
}

extern "C" __device__ void __closesthit__sphere()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Sphere::Data* sphere_data = reinterpret_cast<Sphere::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    const Vec3f world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n.toCUVec()));

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = pgGetSphereUV(local_n);
    si->surface_info = data->surface_info;

    float phi = atan2(local_n.z(), local_n.x());
    if (phi < 0) phi += 2.0f * math::pi;
    const float theta = acos(local_n.y());
    const Vec3f dpdu = Vec3f(-math::two_pi * local_n.z(), 0, math::two_pi * local_n.x());
    const Vec3f dpdv = math::pi * Vec3f(local_n.y() * cos(phi), -sin(theta), local_n.y() * sin(phi));
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu.toCUVec()));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv.toCUVec()));
}

static __forceinline__ __device__ bool hitPlane(
    const Plane::Data* plane_data, const Vec3f& o, const Vec3f& v, const float tmin, const float tmax, SurfaceInteraction& si)
{
    const Vec2f min = plane_data->min;
    const Vec2f max = plane_data->max;

    const float t = -o.y() / v.y();
    const float x = o.x() + t * v.x();
    const float z = o.z() + t * v.z();

    if (min.x() < x && x < max.x() && 
        min.y() < z && z < max.y() && 
        tmin < t && t < tmax)
    {
        si.shading.uv = Vec2f((x - min.x()) / (max.x() - min.x()), 
                      (z - min.y()) / (max.y() - min.y()));
        si.shading.n = Vec3f(0, 1, 0);
        si.t = t;
        si.p = o + t * v;
        return true;
    }
    return false;
}

extern "C" __device__ float __continuation_callable__pdf_plane(const AreaEmitterInfo& area_info, const Vec3f& origin, const Vec3f& direction)
{
    const Plane::Data* plane_data = reinterpret_cast<Plane::Data*>(area_info.shape_data);

    SurfaceInteraction si;
    const Vec3f local_o = area_info.worldToObj.pointMul(origin);
    const Vec3f local_d = area_info.worldToObj.vectorMul(direction);

    if (!hitPlane(plane_data, local_o, local_d, 0.01f, 1e16f, si))
        return 0.0f;

    const Vec3f corner0 = area_info.objToWorld.pointMul(Vec3f(plane_data->min.x(), 0.0f, plane_data->min.y()));
    const Vec3f corner1 = area_info.objToWorld.pointMul(Vec3f(plane_data->max.x(), 0.0f, plane_data->min.y()));
    const Vec3f corner2 = area_info.objToWorld.pointMul(Vec3f(plane_data->min.x(), 0.0f, plane_data->max.y()));
    si.shading.n = normalize(area_info.objToWorld.vectorMul(si.shading.n));
    const float area = length(cross(corner1 - corner0, corner2 - corner0));
    const float distance_squared = si.t * si.t;
    const float cosine = fabs(dot(si.shading.n, direction));
    if (cosine < math::eps)
        return 0.0f;
    return distance_squared / (cosine * area);
}

// グローバル空間における si.p -> 光源上の点 のベクトルを返す
extern "C" __device__ Vec3f DC_FUNC(rnd_sample_plane)(const AreaEmitterInfo& area_info, SurfaceInteraction* si)
{
    const Plane::Data* plane_data = reinterpret_cast<Plane::Data*>(area_info.shape_data);
    // サーフェスの原点をローカル空間に移す
    const Vec3f local_p = area_info.worldToObj.pointMul(si->p);
    unsigned int seed = si->seed;
    // 平面光源上のランダムな点を取得
    const Vec3f rnd_p = Vec3f(rnd(seed, plane_data->min.x(), plane_data->max.x()), 0.0f, rnd(seed, plane_data->min.y(), plane_data->max.y()));
    Vec3f to_light = rnd_p - local_p;
    to_light = area_info.objToWorld.vectorMul(to_light);
    si->seed = seed;
    return to_light;
}

extern "C" __device__ void __intersection__plane()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Plane::Data* plane = reinterpret_cast<Plane::Data*>(data->shape_data);

    const Vec2f min = plane->min;
    const Vec2f max = plane->max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y() / ray.d.y();

    const float x = ray.o.x() + t * ray.d.x();
    const float z = ray.o.z() + t * ray.d.z();

    Vec2f uv((x - min.x()) / (max.x() - min.x()), (z - min.y()) / (max.y() - min.y()));

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, Vec2f_as_ints(uv));
}

extern "C" __device__ void __closesthit__plane()
{
    const HitgroupData* data = reinterpret_cast<const HitgroupData*>(optixGetSbtDataPointer());
    const Plane::Data* plane = reinterpret_cast<Plane::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Vec3f local_n = Vec3f(0, 1, 0);
    const Vec3f world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n.toCUVec()));
    const Vec2f uv = getVec2fFromAttribute<0>();

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = uv;
    si->surface_info = data->surface_info;
    si->shading.dpdu = optixTransformNormalFromObjectToWorldSpace({1.0f, 0.0f, 0.0f});
    si->shading.dpdv = optixTransformNormalFromObjectToWorldSpace({0.0f, 0.0f, 1.0f});
}