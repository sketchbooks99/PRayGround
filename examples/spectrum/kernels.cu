#include <prayground/prayground.h>

#include "params.h"

using namespace prayground;

// Utilities ------------------------------------------------------------------------------
#define SAMPLE_FUNC(name) __direct_callable__sample_ ## name
#define BSDF_FUNC(name) __continuation_callable__bsdf_ ## name
#define PDF_FUNC(name) __direct_callable__pdf_ ## name

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Spectrum>;

static INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = getPayload<0>();
    const unsigned int u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void traceSpectrum(
    OptixTraversableHandle handle, 
    float3 ro, float3 rd, 
    float tmin, float tmax, 
    unsigned int ray_type, 
    SurfaceInteraction* si, 
    float& lambda
)
{
    unsigned int u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle, 
        ro, 
        rd, 
        tmin, 
        tmax, 
        0.0f, 
        OptixVisibilityMask(1), 
        OPTIX_RAY_FLAG_NONE, 
        ray_type, 
        1, 
        ray_type, 
        u0, u1, 
        __float_as_int(lambda)
    );
}

// Raygen ------------------------------------------------------------------------------
static __forceinline__ __device__ void getCameraRay(
    const CameraData& camera, 
    const float& x, const float& y, 
    float3& ro, float3& rd)
{
    rd = normalize(x * camera.U + y * camera.V + camera.W);
    ro = camera.origin;
}

static __forceinline__ __device__ float uniformSpectrumPDF()
{
    return 1.0f / (max_lambda - min_lambda);
}

// Raygen function
extern "C" __global__ void __raygen__spectrum()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int subframe_index = params.subframe_index;
    const uint3 idx = optixGetLaunchIndex();
    unsigned int seed = tea<4>(idx.x * params.width + idx.y, subframe_index);

    float radiance;

    int i = params.samples_per_launch;

    // Uniform sampling of lambda
    const float lambda = lerp(min_lambda, max_lambda, rnd(seed));

    do 
    {
        const float2 subpixel_jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);
        const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(params.width), 
            (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(params.height)
        ) - 1.0f;

        float3 ro, rd;
        getCameraRay(raygen->camera, d.x, d.y, ro, rd);
        
        float throughput = 1.0f;

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = Spectrum{};
        si.albedo = Spectrum{};
        si.trace_terminate = false;
        si.radiance_evaled = false;

        int depth = 0;
        for (;;)
        {
            if (depth >= params.max_depth)
                break;

            traceSpectrum(params.handle, ro, rd, 0.01f, 1e16f, 0, &si, lambda);

            if (si.trace_terminate)
            {
                radiance += si.emission.getSpectrumFromWavelength(lambda) * throughput;
                break;
            }

            // Get emission from area emitter
            if (si.surface_info.type == SurfaceType::AreaEmitter)
            {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id, 
                    &si, 
                    si.surface_info.data
                );
                radiance += si.emission.getSpectrumFromWavelength(lambda) * throughput;
                if (si.trace_terminate)
                    break;
            }
            // Specular sampling
            else if (+(si.surface_info.type & SurfaceType::Delta))
            {
                // Samling scattered direction 
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id, 
                    &si, 
                    si.surface_info.data
                );

                // Evaluate bsdf
                float bsdf_val = optixContinuationCall<float, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id, 
                    &si, 
                    si.surface_info.data
                );

                throughput *= bsdf_val;
            }
            // Rough surface sampling with applying MIS
            else if (+(si.surface_info.type & (SurfaceType::Rough | SurfaceType::Diffuse)))
            {
                unsigned int seed = si.seed;
                AreaEmitterInfo light;
                if (params.num_lights > 0)
                {
                    const int light_id = rnd_int(seed, 0, params.num_lights);
                    light = params.lights[light_id];
                }

                const float weight = 1.0f / (params.num_lights + 1);
                float pdf_val = 0.0f;

                // Importance sampling according to the BSDF
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id, 
                    &si, 
                    si.surface_info.data
                );

                if (rnd(seed) < weight * params.num_lights)
                {
                    // Light sampling
                    float3 to_light = optixDirectCall<float3, AreaEmitterInfo, SurfaceInteraction*>(
                        light.sample_id, 
                        light, 
                        &si
                    );
                    si.wo = normalize(to_light);
                }

                for (int i = 0; i < params.num_lights; i++)
                {
                    // Evaluate PDF of area emitter
                    float light_pdf = optixContinuationCall<float, AreaEmitterInfo, const float3&, const float3&>(
                        params.lights[i].pdf_id,
                        params.lights[i],
                        si.p, 
                        si.wo
                    );
                    pdf_val += weight * light_pdf;
                }

                // Evaluate PDF depends on BSDF 
                float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                    si.surface_info.pdf_id, 
                    &si, 
                    si.surface_info.data
                );

                pdf_val += weight * bsdf_pdf;

                // Evaluate BSDF
                float bsdf_val = optixContinuationCall<float, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id, 
                    &si, 
                    si.surface_info.data
                );

                pdf_val = fmaxf(pdf_val, math::eps);

                throughput *= bsdf_val / pdf_val;
            }

            ro = si.p;
            rd = si.wo;

            ++depth;
        }
    } while (--i);

    const unsigned int image_idx = idx.x * params.width + idx.y;

    float3 xyz_result = make_float3(
        radiance * CIE_X(lambda) / CIE_Y_integral / uniformSpectrumPDF(),
        radiance * CIE_X(lambda) / CIE_Y_integral / uniformSpectrumPDF(),
        radiance * CIE_X(lambda) / CIE_Y_integral / uniformSpectrumPDF()
    );

    float3 color = XYZToSRGB(xyz_result);

    float3 accum_color = color / static_cast<float>(params.samples_per_launch);

    if (subframe_index > 0)
    {
        const float a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_idx]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_idx] = make_float4(accum_color, 1.0f);
    uchar3 ucolor = make_color(accum_color);
    params.result_buffer[image_idx] = make_uchar4(ucolor.x, ucolor.y, ucolor.z, 255);
}

// Miss ------------------------------------------------------------------------------
extern "C" __device__ void __miss__envmap()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    EnvironmentEmitterData* env = reinterpret_cast<EnvironmentEmitterData*>(data->env_data);
    SurfaceInteraction* si = getSurfaceInteraction();

    Ray ray = getWorldRay();
    const float lambda = __int_as_float(getPayload<2>());

    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f*1e8f;
    const float D = half_b * half_b - a * c;

    float sqrtD = sqrtf(D);
    float t = (-half_b + sqrtD) / a;

    const float3 p = normalize(ray.at(t));

    const float phi = atan2(p.z, p.x);
    const float theta = asin(p.y);
    const float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    const float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    si->uv = make_float2(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(
        env->tex_program_id, si, env->tex_data
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
    return sqrtf(1.0f + ((1.03961212f * l2) / (l2 - 0.00600069867f)) + ((0.231792344f * l2) / (l2 - 0.0200179144f)) + ((1.01046945 * l2) / (l2 - 103.560653f)));
}

extern "C" __device__ void SAMPLE_FUNC(dielectric)(const float& lambda, SurfaceInteraction* si, void* mat_data)
{
    const DielectricData* dielectric = reinterpret_cast<DielectricData*>(mat_data);

    float ni = 1.000292f;
    float nt = bk7Index(lambda);
    float cosine = dot(si->wo, si->shading.n);
    bool into = cosine < 0;
    float3 outward_normal = into ? si->shading.n : -si->shading.n;

    if (!into) swap(ni, nt);

    cosine = fabs(cosine);
    float sine = sqrtf(1.0f - cosine * cosine);
    bool cannot_refract = ni * sine > nt;

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

extern "C" __device__ float BSDF_FUNC(dielectric)(const float& lambda, SurfaceInteraction* si, void* mat_data)
{
    const DielectricData* dielectric = reinterpret_cast<DielectricData*>(mat_data);
    si->emission = Spectrum{};  
    Spectrum albedo = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(dielectric->tex_program_id, si, dielectric->tex_data);
    return albedo.getSpectrumFromWavelength(lambda);
}

extern "C" __device__ float PDF_FUNC(dielectric)(SurfaceInteraction * si, void* mat_data)
{
    return 1.0f;
}

extern "C" __device__ void SAMPLE_FUNC(diffuse)(const float& /* lambda */, SurfaceInteraction * si, void* mat_data)
{
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(mat_data);

    if (diffuse->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);
    
    si->trace_terminate = false;
    unsigned int seed = si->seed;
    const float z0 = rnd(seed);
    const float z1 = rnd(seed);
    float3 wi = cosineSampleHemisphere(z0, z1);
    Onb onb(si->shading.n);
    onb.inverseTransform(wi);
    si->wi = normalize(wi);
    si->seed = seed;
}

extern "C" __device__ float BSDF_FUNC(diffuse)(const float& lambda, SurfaceInteraction* si, void* mat_data)
{
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(mat_data);
    const Spectrum albedo = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(diffuse->tex_program_id, si, diffuse->tex_data);
    si->emission = Spectrum{};
    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    return albedo.getSpectrumFromWavelength(lambda) * (cosine / math::pi);
}

extern "C" __device__ float PDF_FUNC(diffuse)(SurfaceInteraction* si, void* mat_data)
{
    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    return cosine / math::pi;
}

extern "C" __device__ void SAMPLE_FUNC(disney)(const float& /* lambda */, SurfaceInteraction * si, void* mat_data)
{
    const DisneyData* disney = reinterpret_cast<DisneyData*>(mat_data);
    if (disney->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    unsigned int seed = si->seed;
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    Onb onb(si->shading.n);

    if (rnd(seed) < diffuse_ratio)
    {
        float3 wi = cosineSampleHemisphere(z1, z2);
        onb.inverseTransform(wi);
        si->wi = normalize(wi);
    }
    else
    {
        float gtr2_ratio = 1.0f / (1.0f + disney->clearcoat);
        float3 h;
        const float alpha = fmaxf(0.001f, disney->roughness);
        if (rnd(seed) < gtr2_ratio)
            h = sampleGGX(z1, z2, alpha);
        else
            h = sampleGTR1(z1, z2, alpha);
        onb.inverseTransform(h);
        si->wi = normalize(reflect(si->wo, h));
    }
    si->radiance_evaled = false;
    si->trace_terminate = false;
    si->seed = seed;
}

extern "C" __device__ float BSDF_FUNC(disney)(const float& lambda, SurfaceInteraction * si, void* mat_data)
{
    const DisneyData* disney = reinterpret_cast<DisneyData*>(mat_data);
    si->emission = Spectrum{};

    const float3 V = -normalize(si->wo);
    const float3 L = normalize(si->wi);
    const float3 N = normalize(si->shading.n);

    const float NdotV = fabs(dot(N, V));
    const float NdotL = fabs(dot(N, L));

    if (NdotV == 0.0f || NdotL == 0.0f)
        return 0.0f;

    const float3 H = normalize(V + L);
    const float NdotH = dot(N, H);
    const float LdotH /* = VdotH */ = dot(L, H);

    const Spectrum base_color = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(
        disney->base_program_id, si, disney->base_tex_data
        );
    const float base_spectrum = base_color.getSpectrumFromWavelength(lambda);

    // Diffuse term (diffuse, subsurface, sheen) ======================
    // Diffuse
    const float Fd90 = 0.5f + 2.0f * disney->roughness * LdotH*LdotH;
    const float FVd90 = fresnelSchlickT(NdotV, Fd90);
    const float FLd90 = fresnelSchlickT(NdotL, Fd90);
    const float f_diffuse = (base_spectrum / math::pi) * FVd90 * FLd90;

    // Subsurface
    const float Fss90 = disney->roughness * LdotH*LdotH;
    const float FVss90 = fresnelSchlickT(NdotV, Fss90);
    const float FLss90 = fresnelSchlickT(NdotL, Fss90); 
    const float f_subsurface = (base_spectrum / math::pi) * 1.25f * (FVss90 * FLss90 * ((1.0f / (NdotV * NdotL)) - 0.5f) + 0.5f);

    // Sheen
    const float lumi = base_color.y() / CIE_Y_integral;
    const float rho_tint = base_spectrum / lumi;
    const float rho_sheen = lerp(1.0f, rho_tint, disney->sheen_tint);
    const float f_sheen = disney->sheen * rho_sheen * powf(1.0f - LdotH, 5.0f);

    // Specular term (specular, clearcoat) ============================
    // Spcular
    const float3 X = si->shading.dpdu;
    const float3 Y = si->shading.dpdv;
    const float alpha = fmaxf(0.001f, disney->roughness);
    const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
    const float ax = fmaxf(0.001f, math::sqr(alpha) / aspect);
    const float ay = fmaxf(0.001f, math::sqr(alpha) * aspect);
    const float rho_specular = lerp(1.0f, rho_tint, disney->specular_tint);
    const float Fs0 = lerp(0.08f * disney->specular * rho_specular, base_spectrum, disney->metallic);
    const float FHs0 = fresnelSchlickR(LdotH, Fs0);
    const float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);
    const float f_specular = FHs0 * Ds * Gs;

    // Clearcoat
    const float Fcc = fresnelSchlickR(LdotH, 0.04f);
    const float alpha_cc = 0.1f + (0.001f - 0.1f) * disney->clearcoat_gloss; // lerp
    const float Dcc = GTR1(NdotH, alpha_cc);
    // const float Gcc = smithG_GGX(N, V, L, 0.25f);
    const float Gcc = smithG_GGX(NdotV, 0.25f);
    const float f_clearcoat = 0.25f * disney->clearcoat * Fcc * Dcc * Gcc;

    const float out = ( 1.0f - disney->metallic ) * ( lerp( f_diffuse, f_subsurface, disney->subsurface ) + f_sheen ) + f_specular + f_clearcoat;
    return out * clamp(NdotL, 0.0f, 1.0f);
}

extern "C" __device__ float PDF_FUNC(disney)(SurfaceInteraction* si, void* mat_data)
{
    const DisneyData* disney = reinterpret_cast<DisneyData*>(mat_data);

    const float3 V = -si->wo;
    const float3 L = si->wi;
    const float3 N = si->shading.n;
    const float3 H = normalize(V + L);

    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    const float specular_ratio = 1.0f - diffuse_ratio;

    const float NdotL = fabs(dot(N, L));
    const float NdotV = fabs(dot(N, V));
    const float NdotH = fabs(dot(H, N));

    const float alpha = fmaxf(0.001f, disney->roughness);
    const float alpha_cc = 0.1f + (0.001f - 0.1f) * disney->clearcoat_gloss;

    const float pdf_Ds = GTR2(NdotH, alpha);
    const float pdf_Dcc = GTR1(NdotH, alpha_cc);
    const float ratio = 1.0f / (1.0f + disney->clearcoat);
    const float pdf_specular = (pdf_Dcc + ratio * (pdf_Ds - pdf_Dcc));
    const float pdf_diffuse = NdotL / math::pi;

    return diffuse_ratio * pdf_diffuse + specular_ratio * pdf_specular;
}

// Area emitter
extern "C" __device__ void __direct_callable__area_emitter(SurfaceInteraction* si, void* surface_data)
{
    const AreaEmitterData* area = reinterpret_cast<AreaEmitterData*>(surface_data);
    si->trace_terminate = true;
    float is_emitted = 1.0f;
    if (!area->twosided)
        is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    
    const Spectrum base = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(
        area->tex_program_id, si, area->tex_data);
    si->albedo = base;
    si->emission = base * area->intensity * is_emitted;
}

// Texture functions
extern "C" __device__ Spectrum DC_FUNC(constant)(SurfaceInteraction * si, void* tex_data)
{
    const ConstantTexture::Data* constant = reinterpret_cast<ConstantTexture::Data*>(tex_data);
    return constant->color;
}

extern "C" __device__ Spectrum DC_FUNC(checker)(SurfaceInteraction * si, void* tex_data)
{
    const CheckerTexture::Data* checker = reinterpret_cast<CheckerTexture::Data*>(tex_data);
    const bool is_odd = sinf(si->uv.x * math::pi * checker->scale) * sinf(si->uv.y * math::pi * checker->scale);
    return is_odd ? checker->color1 : checker->color2;
}

extern "C" __device__ Spectrum DC_FUNC(bitmap)(SurfaceInteraction * si, void* tex_data)
{
    const BitmapTexture::Data* image = reinterpret_cast<BitmapTextureData*>(tex_data);
    float4 c = tex2D<float4>(image->texture, si->uv.x, si->uv.y);
    return Spectrum::fromRGB(make_float3(c.x, c.y, c.z));
}

// Hitgroup functions
extern "C" __device__ void CH_FUNC(mesh)()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const MeshData* mesh = reinterpret_cast<MeshData*>(data->shape_data);

    Ray ray = getWorldRay();

    const int id = optixGetPrimitiveIndex();
    const Face face = mesh->faces[id];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float3 p0 = mesh->vertices[face.vertex_id.x];
    const float3 p1 = mesh->vertices[face.vertex_id.y];
    const float3 p2 = mesh->vertices[face.vertex_id.z];

    const float2 texcoord0 = mesh->texcoords[face.texcoord_id.x];
    const float2 texcoord1 = mesh->texcoords[face.texcoord_id.y];
    const float2 texcoord2 = mesh->texcoords[face.texcoord_id.z];
    const float2 texcoords = (1 - u - v) * texcoord0 + u * texcoord1 + v * texcoord2;

    const float3 n0 = mesh->normals[face.normal_id.x];
    const float3 n1 = mesh->normals[face.normal_id.y];
    const float3 n2 = mesh->normals[face.normal_id.z];

    const float3 local_n = (1 - u - v) * n0 + u * n1 + v * n2;
    const float3 world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n));

    auto si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = texcoords;
    si->surface_info = data->surface_info;

    float3 dpdu, dpdv;
    const float2 duv02 = texcoord0 - texcoord2;
    const float2 duv12 = texcoord1 - texcoord2;
    const float3 dp02 = p0 - p2;
    const float3 dp12 = p1 - p2;
    const float D = duv02.x * duv12.y - duv02.y * duv12.x;
    bool degenerateUV = abs(D) < 1e-8f;
    if (!degenerateUV)
    {
        const float invD = 1.0f / D;
        dpdu = (duv12.y * dp02 - duv02.y * dp12) * invD;
        dpdv = (-duv12.x * dp02 + duv02.x * dp12) * invD;
    }
    if (degenerateUV || length(cross(dpdu, dpdv)) == 0.0f)
    {
        const float3 n = normalize(cross(p2 - p0, p1 - p0));
        Onb onb(n);
        dpdu = onb.tangent;
        dpdv = onb.bitangent;
    }
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
}

static __forceinline__ __device__ float2 getSphereUV(const float3& p) {
    float phi = atan2(p.z, p.x);
    float theta = asin(p.y);
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    return make_float2(u, v);
}

extern "C" __device__ void IS_FUNC(sphere)()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const SphereData* sphere = reinterpret_cast<SphereData*>(data->shape_data);

    const float3 center = sphere->center;
    const float radius = sphere->radius;
    
    Ray ray = getLocalRay();

    const float3 oc = ray.o - center;
    const float a = dot(ray.d, ray.d);
    const float half_b = dot(oc, ray.d);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant > 0.0f) {
        float sqrtd = sqrtf(discriminant);
        float t1 = (-half_b - sqrtd) / a;
        bool check_second = true;
        if (t1 > ray.tmin && t1 < ray.tmax) {
            float3 normal = normalize((ray.at(t1) - center) / radius);
            check_second = false;
            optixReportIntersection(t1, 0, float3_as_ints(normal));
        }

        if (check_second) {
            float t2 = (-half_b + sqrtd) / a;
            if (t2 > ray.tmin && t2 < ray.tmax) {
                float3 normal = normalize((ray.at(t2) - center) / radius);
                optixReportIntersection(t2, 0, float3_as_ints(normal));
            }
        }
    }
}

extern "C" __device__ void CH_FUNC(sphere)()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const SphereData* sphere_data = reinterpret_cast<SphereData*>(data->shape_data);

    Ray ray = getWorldRay();

    float3 local_n = getFloat3FromAttribute<0>();
    const float3 world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n));

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = getSphereUV(local_n);
    si->surface_info = data->surface_info;

    float phi = atan2(local_n.z, local_n.x);
    if (phi < 0) phi += 2.0f * math::pi;
    const float theta = acos(local_n.y);
    const float3 dpdu = make_float3(-math::two_pi * local_n.z, 0, math::two_pi * local_n.x);
    const float3 dpdv = math::pi * make_float3(local_n.y * cos(phi), -sin(theta), local_n.y * sin(phi));
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
}

extern "C" __device__ void IS_FUNC(plane)()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const PlaneData* plane_data = reinterpret_cast<PlaneData*>(data->shape_data);

    const float2 min = plane_data->min;
    const float2 max = plane_data->max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y / ray.d.y;

    const float x = ray.o.x + t * ray.d.x;
    const float z = ray.o.z + t * ray.d.z;

    float2 uv = make_float2((x - min.x) / (max.x - min.x), (z - min.y) / (max.y - min.y));

    if (min.x < x && x < max.x && min.y < z && z < max.y && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, float2_as_ints(uv));
}

extern "C" __device__ void CH_FUNC(plane)()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    float3 local_n = make_float3(0, 1, 0);
    const float3 world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n));
    const float2 uv = getFloat2FromAttribute<0>();

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;
    si->shading.dpdu = optixTransformNormalFromObjectToWorldSpace(make_float3(1.0f, 0.0f, 0.0f));
    si->shading.dpdv = optixTransformNormalFromObjectToWorldSpace(make_float3(0.0f, 0.0f, 1.0f));
}