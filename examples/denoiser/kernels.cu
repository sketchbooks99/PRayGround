#include <prayground/prayground.h>

#include "params.h"

using namespace prayground;

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

INLINE DEVICE void trace(
    OptixTraversableHandle handle,
    const Vec3f& ro, const Vec3f& rd,
    float tmin, float tmax,
    uint32_t ray_type,
    uint32_t ray_count,
    SurfaceInteraction* si
)
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle,
        ro, rd,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        ray_type,
        1,
        ray_type,
        u0, u1
    );
}

// --------------------------------------------------------------------------
// Raygen 
// --------------------------------------------------------------------------
static __forceinline__ __device__ void getCameraRay(const Camera::Data& camera, float x, float y, Vec3f& ro, Vec3f& rd)
{
    rd = normalize(x * camera.U + y * camera.V + camera.W);
    ro = camera.origin;
}

static __forceinline__ __device__ Vec3f reinhardToneMap(const Vec3f& color, const float white)
{
    const float l = luminance(color);
    return (color * 1.0f) / (1.0f + l / white);
}

extern "C" __device__ void __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int frame = params.frame;
    const Vec3ui idx(optixGetLaunchIndex());
    unsigned int seed = tea<4>(idx.y() * params.width + idx.x(), frame);

    Vec3f result(0.0f);
    Vec3f normal(0.0f);
    Vec3f albedo(0.0f);

    int spl = params.samples_per_launch;

    for (int i = 0; i < spl; i++)
    {
        const Vec2f jitter = UniformSampler::get2D(seed) - 0.5f;
        const Vec2f res(params.width, params.height);
        const Vec2f d = 2.0f * ((Vec2f(idx.x(), idx.y()) + jitter) / res) - 1.0f;

        Vec3f ro, rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

        Vec3f throughput(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = Vec3f(0.0f);
        si.albedo = Vec3f(0.0f);
        si.shading.n = Vec3f(0.0f);
        si.trace_terminate = false;
        si.radiance_evaled = false;

        float tmax = raygen->camera.farclip / dot(rd, normalize(raygen->camera.W));

        int depth = 0;

        while (true)
        {
            if (depth >= params.max_depth)
                break;

            trace(params.handle, ro, rd, 0.01f, tmax, /* ray_type = */ 0, 1, &si);

            if (si.trace_terminate)
            {
                result += throughput * si.emission;
                break;
            }

            // Get emission from area emitter
            if (si.surface_info.type == SurfaceType::AreaEmitter)
            {
                // Evaluate emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id, &si, si.surface_info.data);
                result += throughput * si.emission;
                
                if (depth == 0)
                {
                    albedo += si.albedo;
                    normal += si.shading.n;
                }

                if (si.trace_terminate)
                    break;
            }
            // Specular sampling
            else if (+(si.surface_info.type & SurfaceType::Delta))
            {
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id, &si, si.surface_info.data);

                // Evaluate BSDF
                const Vec3f bsdf_val = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id, &si, si.surface_info.data);
                throughput *= bsdf_val;
            }
            // Rough surface sampling
            else if (+(si.surface_info.type & (SurfaceType::Rough | SurfaceType::Diffuse)))
            {
                uint32_t seed = si.seed;
                AreaEmitterInfo light;
                if (params.num_lights > 0) {
                    const int light_id = rnd_int(seed, 0, params.num_lights - 1);
                    light = params.lights[light_id];
                }

                const float weight = 1.0f / (params.num_lights + 1);

                float pdf_val = 0.0f;

                // Importance sampling according to the BSDF
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id, &si, si.surface_info.data);

                if (rnd(seed) < weight * params.num_lights) {
                    // Light sampling
                    Vec3f to_light = optixDirectCall<Vec3f, AreaEmitterInfo, SurfaceInteraction*>(
                        light.sample_id, light, &si);
                    si.wi = normalize(to_light);
                }

                for (int i = 0; i < params.num_lights; i++)
                {
                    // Evaluate PDF of area emitter
                    float light_pdf = optixContinuationCall<float, AreaEmitterInfo, const Vec3f&, const Vec3f&>(
                        params.lights[i].pdf_id, params.lights[i], si.p, si.wi);
                    pdf_val += weight * light_pdf;
                }

                // Evaluate PDF depends on BSDF
                float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                    si.surface_info.pdf_id, &si, si.surface_info.data);

                pdf_val += weight * bsdf_pdf;

                // Evaluate BSDF
                Vec3f bsdf_val = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id, &si, si.surface_info.data);

                if (pdf_val == 0.0f) break;

                throughput *= bsdf_val / pdf_val;
            }

            if (depth == 0)
            {
                albedo += si.albedo;
                normal += si.shading.n;
            }

            tmax = 1e16f;

            ro = si.p;
            rd = si.wi;

            depth++;
        }
    }

    const unsigned int image_idx = idx.y * params.width + idx.x;

    result.x() = isnan(result.x()) || isinf(result.x()) ? 0.0f : result.x();
    result.y() = isnan(result.y()) || isinf(result.y()) ? 0.0f : result.y();
    result.z() = isnan(result.z()) || isinf(result.z()) ? 0.0f : result.z();

    Vec3f accum = result / (float)spl;
    albedo = albedo / (float)spl;
    normal = normal / (float)spl;

    if (frame > 0)
    {
        const float a = 1.0f / (float)(frame + 1);
        const Vec3f accum_prev(params.accum_buffer[image_idx]);
        accum = lerp(accum_prev, accum, a);
        const Vec3f albedo_prev(params.albedo_buffer[image_idx]);
        const Vec3f normal_prev(params.normal_buffer[image_idx]);
        albedo = lerp(albedo_prev, albedo / (float)spl, a);
        normal = lerp(normal_prev, normal / (float)spl, a);
    }
    params.accum_buffer[image_idx] = Vec4f(accum, 1.0f);
    Vec3u color = make_color(accum);
    params.result_buffer[image_idx] = Vec4f(color2float(color), 1.0f);
    params.normal_buffer[image_idx] = Vec4f(normal, 1.0f);
    params.albedo_buffer[image_idx] = Vec4f(albedo, 1.0f);
}

// --------------------------------------------------------------------------
// Miss
// --------------------------------------------------------------------------
extern "C" __device__ void __miss__envmap()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    EnvironmentEmitter::Data* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
    SurfaceInteraction* si = getSurfaceInteraction();

    Ray ray = getWorldRay();

    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f * 1e8f;
    const float discriminant = half_b * half_b - a * c;

    float sqrtd = sqrtf(discriminant);
    float t = (-half_b + sqrtd) / a;

    Vec3f p = normalize(ray.at(t));

    float phi = atan2(p.z, p.x);
    float theta = asin(p.y);
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) * math::inv_pi;
    si->uv = Vec2f(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data;
}

// --------------------------------------------------------------------------
// Callables for surface
// --------------------------------------------------------------------------
// Diffuse -----------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_diffuse(SurfaceInteraction* si, void* mat_data) {
    const auto* diffuse = reinterpret_cast<Diffuse::Data*>(mat_data);

    if (diffuse->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    si->trace_terminate = false;
    uint32_t seed = si->seed;
    const Vec2f u = UniformSampler::get2D(seed);
    Vec3f wi = cosineSampleHemisphere(u[0], u[1]);
    Onb onb(si->shading.n);
    onb.inverseTransform(wi);
    si->wi = normalize(wi);
    si->seed = seed;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
    const auto* diffuse = reinterpret_cast<Diffuse::Data*>(mat_data);
    const Vec4f albedo = optixDirectCall<Vec4f, SurfaceInteraction*, void*>(diffuse->texture.prg_id, si, diffuse->texture.data);
    si->albedo = Vec3f(albedo) * albedo.w;
    si->emission = Vec3f(0.0f);
    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    return si->albedo * cosine * math::inv_pi;
}

extern "C" __device__ float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    return cosine * math::inv_pi;
}

// Dielectric --------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_dielectric(SurfaceInteraction* si, void* mat_data) {
    const auto* dielectric = reinterpret_cast<Dielectric::Data*>(mat_data);

    float ni = 1.000292f; // air
    float nt = dielectric->ior;  // ior specified 
    float cosine = dot(si->wi, si->shading.n);
    bool into = cosine < 0;
    Vec3f outward_normal = into ? si->shading.n : -si->shading.n;

    if (!into) swap(ni, nt);

    cosine = fabs(cosine);
    float sine = sqrtf(1.0 - cosine*cosine);
    bool cannot_refract = (ni / nt) * sine > 1.0f;

    float reflect_prob = fresnel(cosine, ni, nt);
    unsigned int seed = si->seed;

    if (cannot_refract || reflect_prob > rnd(seed))
        si->wi = reflect(si->wo, outward_normal);
    else    
        si->wi = refract(si->wo, outward_normal, cosine, ni, nt);
    si->radiance_evaled = false;
    si->trace_terminate = false;
    si->seed = seed;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_dielectric(SurfaceInteraction* si, void* mat_data)
{
    const auto* dielectric = reinterpret_cast<Dielectric::Data*>(mat_data);
    si->emission = Vec3f(0.0f);
    Vec4f albedo = optixDirectCall<Vec4f, SurfaceInteraction*, void*>(dielectric->texture.prg_id, si, dielectric->texture.data);
    si->albedo = Vec3f(albedo);
    return si->albedo;
}

extern "C" __device__ float __direct_callable__pdf_dielectric(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}

// Conductor --------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_conductor(SurfaceInteraction* si, void* mat_data) {
    const auto* conductor = reinterpret_cast<Conductor::Data*>(mat_data);
    if (conductor->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    si->wi = reflect(si->wo, si->shading.n);
    si->trace_terminate = false;
    si->radiance_evaled = false;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_conductor(SurfaceInteraction* si, void* mat_data)
{
    const auto* conductor = reinterpret_cast<Conductor::Data*>(mat_data);
    si->emission = Vec3f(0.0f);
    Vec4f albedo = optixDirectCall<Vec4f, SurfaceInteraction*, void*>(conductor->texture.prg_id, si, conductor->texture.data);
    si->albedo = Vec3f(albedo);
    return si->albedo;
}

extern "C" __device__ float __direct_callable__pdf_conductor(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}

// Disney BRDF ------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_disney(SurfaceInteraction* si, void* mat_data)
{
    const DisneyData* disney = reinterpret_cast<DisneyData*>(mat_data);

    if (disney->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wi, si->shading.n);

    unsigned int seed = si->seed;
    const Vec2f u = UniformSampler::get2D(seed);
    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    Onb onb(si->shading.n);

    if (rnd(seed) < diffuse_ratio)
    {
        Vec3f wi = cosineSampleHemisphere(u[0], u[1]);
        onb.inverseTransform(wi);
        si->wi = normalize(wi);
    }
    else
    {
        float gtr2_ratio = 1.0f / (1.0f + disney->clearcoat);
        Vec3f h;
        const float alpha = fmaxf(0.001f, disney->roughness);
        const float alpha_cc = lerp(0.1f, 0.001f, disney->clearcoat_gloss);
        if (rnd(seed) < gtr2_ratio)
            h = sampleGGX(u[0], u[1], alpha);
        else
            h = sampleGTR1(u[0], u[1], alpha_cc);
        onb.inverseTransform(h);
        si->wi = normalize(reflect(si->wo, h));
    }
    si->radiance_evaled = false;
    si->trace_terminate = false;
    si->seed = seed;
}

/**
 * @ref: https://rayspace.xyz/CG/contents/Disney_principled_BRDF/
 * 
 * @note 
 * ===== Prefix =====
 * F : fresnel 
 * f : brdf function
 * G : geometry function
 * D : normal distribution function
 */
extern "C" __device__ Vec3f __continuation_callable__bsdf_disney(SurfaceInteraction* si, void* mat_data)
{   
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);
    si->emission = Vec3f(0.0f);

    const Vec3f V = -normalize(si->wo);
    const Vec3f L = normalize(si->wi);
    const Vec3f N = normalize(si->shading.n);

    const float NdotV = dot(N, V);
    const float NdotL = dot(N, L);

    if (NdotV <= 0.0f || NdotL <= 0.0f)
        return Vec3f(0.0f);

    const Vec3f H = normalize(V + L);
    const float NdotH = dot(N, H);
    const float LdotH /* = VdotH */ = dot(L, H);

    const Vec4f base_color = optixDirectCall<Vec4f, SurfaceInteraction*, void*>(
        disney->base.prg_id, si, disney->base.data);
    si->albedo = base_color;

    // Diffuse term (diffuse, subsurface, sheen) ======================
    // Diffuse
    const float Fd90 = 0.5f + 2.0f * disney->roughness * LdotH*LdotH;
    const float FVd90 = fresnelSchlickT(NdotV, Fd90);
    const float FLd90 = fresnelSchlickT(NdotL, Fd90);
    const Vec3f f_diffuse = base_color * math::inv_pi * FVd90 * FLd90;

    // Subsurface
    const float Fss90 = disney->roughness * LdotH*LdotH;
    const float FVss90 = fresnelSchlickT(NdotV, Fss90);
    const float FLss90 = fresnelSchlickT(NdotL, Fss90); 
    const Vec3f f_subsurface = base_color * math::inv_pi * 1.25f * (FVss90 * FLss90 * ((1.0f / (NdotV * NdotL)) - 0.5f) + 0.5f);

    // Sheen
    const Vec3f rho_tint = base_color / luminance(base_color);
    const Vec3f rho_sheen = lerp(Vec3f(1.0f), rho_tint, disney->sheen_tint);
    const Vec3f f_sheen = disney->sheen * rho_sheen * powf(1.0f - LdotH, 5.0f);

    // Specular term (specular, clearcoat) ============================
    // Spcular
    const Vec3f X = si->shading.dpdu;
    const Vec3f Y = si->shading.dpdv;
    const float alpha = fmaxf(0.001f, disney->roughness);
    const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
    const float ax = fmaxf(0.001f, pow2(alpha) / aspect);
    const float ay = fmaxf(0.001f, pow2(alpha) * aspect);
    const Vec3f rho_specular = lerp(Vec3f(1.0f), rho_tint, disney->specular_tint);
    const Vec3f Fs0 = lerp(0.08f * disney->specular * rho_specular, base_color, disney->metallic);
    const Vec3f FHs0 = fresnelSchlickR(LdotH, Fs0);
    const float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);
    const Vec3f f_specular = FHs0 * Ds * Gs;

    // Clearcoat
    const float Fcc = fresnelSchlickR(LdotH, 0.04f);
    const float alpha_cc = lerp(0.1f, 0.001f, disney->clearcoat_gloss);
    const float Dcc = GTR1(NdotH, alpha_cc);
    const float Gcc = smithG_GGX(NdotV, 0.25f);
    const Vec3f f_clearcoat = Vec3f( 0.25f * disney->clearcoat * (Fcc * Dcc * Gcc) );

    const Vec3f out = ( 1.0f - disney->metallic ) * ( lerp( f_diffuse, f_subsurface, disney->subsurface ) + f_sheen ) + f_specular + f_clearcoat;
    return out * clamp(NdotL, 0.0f, 1.0f);
}

/**
 * @ref http://simon-kallweit.me/rendercompo2015/report/#adaptivesampling
 * 
 * @todo Investigate correct evaluation of PDF.
 */
extern "C" __device__ float __direct_callable__pdf_disney(SurfaceInteraction* si, void* mat_data)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);

    const Vec3f V = -si->wo;
    const Vec3f L = si->wi;
    const Vec3f N = si->shading.n;

    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    const float specular_ratio = 1.0f - diffuse_ratio;

    const float NdotL = dot(N, L);
    const float NdotV = dot(N, V);

    if (NdotL <= 0.0f || NdotV <= 0.0f)
        return 1.0f;

    const float alpha = fmaxf(0.001f, disney->roughness);
    const float alpha_cc = lerp(0.1f, 0.001f, disney->clearcoat_gloss);
    const Vec3f H = normalize(V + L);
    const float NdotH = abs(dot(H, N));

    const float pdf_Ds = GTR2(NdotH, alpha);
    const float pdf_Dcc = GTR1(NdotH, alpha_cc);
    const float ratio = 1.0f / (1.0f + disney->clearcoat);
    const float pdf_specular = (pdf_Dcc + ratio * (pdf_Ds - pdf_Dcc));
    const float pdf_diffuse = NdotL / math::pi;

    return diffuse_ratio * pdf_diffuse + specular_ratio * pdf_specular;
}

// Area emitter ------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__area_emitter(SurfaceInteraction* si, void* surface_data)
{
    const AreaEmitter::Data* area = reinterpret_cast<AreaEmitter::Data*>(surface_data);
    si->trace_terminate = true;
    float is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    if (area->twosided)
    {
        is_emitted = 1.0f;
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);
    }

    // Get texture
    auto color = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(
        area->texture.prg_id, si, area->texture.data);
    si->albedo = Vec3f(base);
    
    si->emission = si->albedo * area->intensity * is_emitted;
}

// --------------------------------------------------------------------------
// Callables for texture
// --------------------------------------------------------------------------
extern "C" __device__ Vec4f __direct_callable__bitmap(SurfaceInteraction* si, void* tex_data) {
    const auto* image = reinterpret_cast<BitmapTexture::Data*>(tex_data);
    float4 c = tex2D<float4>(image->texture, si->uv.x(), si->uv.y());
    return c;
}

extern "C" __device__ Vec4f __direct_callable__constant(SurfaceInteraction* si, void* tex_data) {
    const auto* constant = reinterpret_cast<ConstantTexture::Data*>(tex_data);
    return constant->color;
}

extern "C" __device__ Vec4f __direct_callable__checker(SurfaceInteraction* si, void* tex_data) {
    const auto* checker = reinterpret_cast<CheckerTexture::Data*>(tex_data);
    const bool is_odd = sinf(si->uv.x() * math::pi * checker->scale) * sinf(si->uv.y() * math::pi * checker->scale) < 0;
    return lerp(checker->color1, checker->color2, static_cast<float>(is_odd));
}

// --------------------------------------------------------------------------
// Hitgroups
// --------------------------------------------------------------------------
// Plane -------------------------------------------------------------------------------
static __forceinline__ __device__ bool hitPlane(
    const Plane::Data* plane, const Vec3f& o, const Vec3f& v, const float tmin, const float tmax, SurfaceInteraction& si)
{
    const Vec2f min = plane->min;
    const Vec2f max = plane->max;
    
    const float t = -o.y() / v.y();
    const float x = o.x() + t * v.x();
    const float z = o.z() + t * v.z();

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && tmin < t && t < tmax)
    {
        si.uv = Vec2f((x - min.x()) / (max.x() - min.x()), (z - min.y()) / max.y() - min.y());
        si.n = Vec3f(0, 1, 0);
        si.t = t;
        si.p = o + t*v;
        return true;
    }
    return false;
}

extern "C" __device__ void __intersection__plane()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* plane = reinterpret_cast<Plane::Data*>(data->shape_data);

    const Vec2f min = plane->min;
    const Vec2f max = plane->max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y() / ray.d.y();

    const float x = ray.o.x() + t * ray.d.x();
    const float z = ray.o.z() + t * ray.d.z();

    Vec2f uv = Vec2f((x - min.x()) / (max.x() - min.x()), (z - min.y()) / (max.y() - min.y()));

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, Vec2f_as_ints(uv));
}

extern "C" __device__ void __closesthit__plane()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    Vec3f local_n = Vec3f(0, 1, 0);
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);
    Vec2f uv = getVec2fFromAttribute<0>();

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;

    si->dpdu = optixTransformNormalFromObjectToWorldSpace(Vec3f(1, 0, 0));
    si->dpdv = optixTransformNormalFromObjectToWorldSpace(Vec3f(0, 0, 1));
}

extern "C" __device__ float __continuation_callable__pdf_plane(
    const AreaEmitterInfo& area_info, const Vec3f & origin, const Vec3f & direction)
{
    const auto* plane = reinterpret_cast<Plane::Data*>(area_info.shape_data);

    SurfaceInteraction si;
    const Vec3f local_o = area_info.worldToObj.pointMul(origin);
    const Vec3f local_d = area_info.worldToObj.vectorMul(direction);

    if (!hitPlane(plane_data, local_o, local_d, 0.01f, 1e16f, si))
        return 0.0f;

    const Vec3f corner0 = area_info.objToWorld.pointMul(Vec3f(plane_data->min.x(), 0.0f, plane_data->min.y()));
    const Vec3f corner1 = area_info.objToWorld.pointMul(Vec3f(plane_data->max.x(), 0.0f, plane_data->min.y()));
    const Vec3f corner2 = area_info.objToWorld.pointMul(Vec3f(plane_data->min.x(), 0.0f, plane_data->max.y()));
    si.shading.n = normalize(area_info.objToWorld.vectorMul(si.n));
    const float area = length(cross(corner1 - corner0, corner2 - corner0));
    const float distance_squared = si.t * si.t;
    const float cosine = fabs(dot(si.shading.n, direction));
    if (cosine < math::eps)
        return 0.0f;
    return distance_squared / (cosine * area);
}

// Return light vector in global space from si.p to random light point
extern "C" __device__ Vec3f __direct_callable__rnd_sample_plane(AreaEmitterInfo area_info, SurfaceInteraction * si)
{
    const auto* plane = reinterpret_cast<Plane::Data*>(area_info.shape_data);
    // Transform point from world to object space
    const Vec3f local_p = area_info.worldToObj.pointMul(si->p);
    uint32_t seed = si->seed;
    // Get random point on area emitter
    const Vec3f rnd_p(rnd(seed, plane_data->min.x(), plane_data->max.x()), 0.0f, rnd(seed, plane_data->min.y(), plane_data->max.y()));
    Vec3f to_light = rnd_p - local_p;
    to_light = area_info.objToWorld.vectorMul(to_light);
    si->seed = seed;
    return to_light;
}

// Sphere -------------------------------------------------------------------------------
static __forceinline__ __device__ Vec2f getSphereUV(const Vec3f& p) {
    float phi = atan2(p.z(), p.x());
    float theta = asin(p.y());
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) * math::inv_pi;
    return Vec2f(u, v);
}

static __forceinline__ __device__ bool hitSphere(const SphereData* sphere_data, const Vec3f& o, const Vec3f& v, const float tmin, const float tmax, SurfaceInteraction& si)
{
    const Vec3f center = sphere_data->center;
    const float radius = sphere_data->radius;

    const Vec3f oc = o - center;
    const float a = dot(v, v);
    const float half_b = dot(oc, v);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant <= 0.0f) return false;

    const float sqrtd = sqrtf(discriminant);

    float t = (-half_b - sqrtd) / a;
    if (t < tmin || tmax < t)
    {
        t = (-half_b + sqrtd) / a;
        if (t < tmin || tmax < t)
            return false;
    }

    si.t = t;
    si.p = o + t * v;
    si.n = si.p / radius;
    si.uv = getSphereUV(si.shading.n);
    return true;
}

extern "C" __device__ void __intersection__sphere() {
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

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

extern "C" __device__ void __closesthit__sphere() {
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = getSphereUV(local_n);
    si->surface_info = data->surface_info;

    float phi = atan2(local_n.z(), local_n.x());
    if (phi < 0) phi += 2.0f * math::pi;
    const float theta = acos(local_n.y());
    const float u = phi / (2.0f * math::pi);
    const float v = theta / math::pi;
    const Vec3f dpdu = Vec3f(-math::two_pi * local_n.z(), 0, math::two_pi * local_n.x());
    const Vec3f dpdv = math::pi * Vec3f(local_n.y * cos(phi), -sin(theta), local_n.y * sin(phi));
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
}

// Cylinder -------------------------------------------------------------------------------
static INLINE DEVICE Vec2f getCylinderUV(
    const Vec3f& p, const float radius, const float height, const bool hit_disk)
{
    if (hit_disk)
    {
        const float r = sqrtf(p.x()*p.x() + p.z()*p.z()) / radius;
        const float theta = atan2(p.z(), p.x());
        float u = 1.0f - (theta + math::pi / 2.0f) * math::inv_pi;
        return Vec2f(u, r);
    } 
    else
    {
        float phi = atan2(p.z(), p.x());
        if (phi < 0.0f) phi += math::two_pi;
        const float u = phi / math::two_pi;
        const float v = (p.y + height / 2.0f) / height;
        return Vec2f(u, v);
    }
}

extern "C" __device__ void __intersection__cylinder()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* cylinder = reinterpret_cast<Cylinder::Data*>(data->shape_data);

    const float radius = cylinder->radius;
    const float height = cylinder->height;

    Ray ray = getLocalRay();
    SurfaceInteraction* si = getSurfaceInteraction();
    
    const float a = dot(ray.d, ray.d) - ray.d.y() * ray.d.y();
    const float half_b = (ray.o.x() * ray.d.x() + ray.o.z() * ray.d.z());
    const float c = dot(ray.o, ray.o) - ray.o.y() * ray.o.y() - radius*radius;
    const float discriminant = half_b*half_b - a*c;

    if (discriminant > 0.0f)
    {
        const float sqrtd = sqrtf(discriminant);
        const float side_t1 = (-half_b - sqrtd) / a;
        const float side_t2 = (-half_b + sqrtd) / a;

        const float side_tmin = fmin( side_t1, side_t2 );
        const float side_tmax = fmax( side_t1, side_t2 );

        if ( side_tmin > ray.tmax || side_tmax < ray.tmin )
            return;

        const float upper = height / 2.0f;
        const float lower = -height / 2.0f;
        const float y_tmin = fmin( (lower - ray.o.y()) / ray.d.y(), (upper - ray.o.y()) / ray.d.y() );
        const float y_tmax = fmax( (lower - ray.o.y()) / ray.d.y(), (upper - ray.o.y()) / ray.d.y() );

        float t1 = fmax(y_tmin, side_tmin);
        float t2 = fmin(y_tmax, side_tmax);
        if (t1 > t2 || (t2 < ray.tmin) || (t1 > ray.tmax))
            return;
        
        bool check_second = true;
        if (ray.tmin < t1 && t1 < ray.tmax)
        {
            Vec3f P = ray.at(t1);
            bool hit_disk = y_tmin > side_tmin;
            Vec3f normal = hit_disk 
                         ? normalize(P - Vec3f(P.x(), 0.0f, P.z()))   // Hit at disk
                         : normalize(P - Vec3f(0.0f, P.y(), 0.0f));   // Hit at side
            Vec2f uv = getCylinderUV(P, radius, height, hit_disk);
            if (hit_disk)
            {
                const float rHit = sqrtf(P.x()*P.x() + P.z()*P.z());
                si->dpdu = Vec3f(-math::two_pi * P.y(), 0.0f, math::two_pi * P.z());
                si->dpdv = Vec3f(P.x(), 0.0f, P.z()) * radius / rHit;
            }
            else 
            {
                si->dpdu = Vec3f(-math::two_pi * P.z(), 0.0f, math::two_pi * P.x());
                si->dpdv = Vec3f(0.0f, height, 0.0f);
            }
            optixReportIntersection(t1, 0, Vec3f_as_ints(normal), Vec2f_as_ints(uv));
            check_second = false;
        }
        
        if (check_second)
        {
            if (ray.tmin < t2 && t2 < ray.tmax)
            {
                Vec3f P = ray.at(t2);
                bool hit_disk = y_tmax < side_tmax;
                Vec3f normal = hit_disk
                            ? normalize(P - Vec3f(P.x(), 0.0f, P.z()))   // Hit at disk
                            : normalize(P - Vec3f(0.0f, P.y(), 0.0f));   // Hit at side
                Vec2f uv = getCylinderUV(P, radius, height, hit_disk);
                if (hit_disk)
                {
                    const float rHit = sqrtf(P.x()*P.x() + P.z()*P.z());
                    si->shading.dpdu = Vec3f(-math::two_pi * P.y(), 0.0f, math::two_pi * P.z());
                    si->shading.dpdv = Vec3f(P.x(), 0.0f, P.z()) * radius / rHit;
                }
                else 
                {
                    si->shading.dpdu = Vec3f(-math::two_pi * P.z(), 0.0f, math::two_pi * P.x());
                    si->shading.dpdv = Vec3f(0.0f, height, 0.0f);
                }
                optixReportIntersection(t2, 0, Vec3f_as_ints(normal), Vec2f_as_ints(uv));
            }
        }
    }
}

extern "C" __device__ void __closesthit__cylinder()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* cylinder = reinterpret_cast<Cylinder::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec2f uv = getVec2fFromAttribute<3>();

    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shdaing.n = normalize(world_n);
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;

    // dpdu and dpdv are calculated in intersection shader
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(si->dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(si->dpdv));
}

// Triangle mesh -------------------------------------------------------------------------------
extern "C" __device__ void __closesthit__mesh()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* mesh_data = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    Ray ray = getWorldRay();
    
    const int prim_id = optixGetPrimitiveIndex();
    const Face face = mesh_data->faces[prim_id];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const Vec3f p0 = mesh_data->vertices[face.vertex_id.x()];
    const Vec3f p1 = mesh_data->vertices[face.vertex_id.y()];
    const Vec3f p2 = mesh_data->vertices[face.vertex_id.z()];

    const Vec2f texcoord0 = mesh_data->texcoords[face.texcoord_id.x()];
    const Vec2f texcoord1 = mesh_data->texcoords[face.texcoord_id.y()];
    const Vec2f texcoord2 = mesh_data->texcoords[face.texcoord_id.z()];
    const Vec2f texcoords = (1-u-v)*texcoord0 + u*texcoord1 + v*texcoord2;

    const Vec3f n0 = mesh_data->normals[face.normal_id.x()];
	const Vec3f n1 = mesh_data->normals[face.normal_id.y()];
	const Vec3f n2 = mesh_data->normals[face.normal_id.z()];

    // Linear interpolation of normal by barycentric coordinates.
    Vec3f local_n = (1.0f-u-v)*n0 + u*n1 + v*n2;
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = texcoords;
    si->surface_info = data->surface_info;

    // Calculate partial derivative on texture coordinates
    Vec3f dpdu, dpdv;
    const Vec2f duv02 = texcoord0 - texcoord2, duv12 = texcoord1 - texcoord2;
    const Vec3f dp02 = p0 - p2, dp12 = p1 - p2;
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
    si->shading.dpdu = normalize(optixTransformNormalFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformNormalFromObjectToWorldSpace(dpdv));
}

