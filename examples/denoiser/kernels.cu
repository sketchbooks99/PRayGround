#include <prayground/prayground.h>

#include "params.h"

using namespace prayground;

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Spectrum>;

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

INLINE DEVICE void trace(
    OptixTraversableHandle handle,
    const float3& ro, const float3& rd,
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
static __forceinline__ __device__ void getCameraRay(const Camera::Data& camera, float x, float y, float3& ro, float3& rd)
{
    rd = normalize(x * camera.U + y * camera.V + camera.W);
    ro = camera.origin;
}

static __forceinline__ __device__ float3 reinhardToneMap(const float3& color, const float white)
{
    const float l = luminance(color);
    return (color * 1.0f) / (1.0f + l / white);
}

static __forceinline__ __device__ float3 exposureToneMap(const float3& color, const float exposure)
{
    return make_float3(1.0f) - expf(-color * exposure);
}

extern "C" __device__ void __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int frame = params.frame;
    const uint3 idx = optixGetLaunchIndex();
    unsigned int seed = tea<4>(idx.x * params.width + idx.y, frame);

    float3 result = make_float3(0.0f);
    float3 normal = make_float3(0.0f);
    float3 albedo = make_float3(0.0f);

    int spl = params.samples_per_launch;

    for (int i = 0; i < spl; i++)
    {
        const float2 jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

        const float2 d = 2.0f * make_float2(
            ((float)idx.x + jitter.x) / (float)params.width, 
            ((float)idx.y + jitter.y) / (float)params.height
        ) - 1.0f;

        float3 ro, rd;
        getCameraRay(raygen->camera, d.x, d.y, ro, rd);

        float3 throughput = make_float3(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = make_float3(0.0f);
        si.albedo = make_float3(0.0f);
        si.shading.n = make_float3(0.0f);
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
                    si.surface_info.bsdf_id, 
                    &si,
                    si.surface_info.data
                );
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
                    si.surface_info.sample_id,
                    &si,
                    si.surface_info.data
                );

                // Evaluate BSDF
                const float3 bsdf_val = optixContinuationCall<float3, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id,
                    &si,
                    si.surface_info.data
                );
                throughput *= bsdf_val;
            }
            // Rough surface sampling
            else if (+(si.surface_info.type & (SurfaceType::Rough | SurfaceType::Diffuse)))
            {
                unsigned int seed = si.seed;
                AreaEmitterInfo light;
                if (params.num_lights > 0) {
                    const int light_id = rnd_int(seed, 0, params.num_lights - 1);
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

                if (rnd(seed) < weight * params.num_lights) {
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
                float3 bsdf_val = optixContinuationCall<float3, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id,
                    &si,
                    si.surface_info.data
                    );

                if (pdf_val == 0.0f) break;

                throughput *= bsdf_val / pdf_val;
            }

            if (depth == 0)
            {
                albedo += si.albedo;
                normal += si.n;
            }

            tmax = 1e16f;

            ro = si.p;
            rd = si.wo;

            depth++;
        }
    }

    const unsigned int image_idx = idx.y * params.width + idx.x;

    result.x = isnan(result.x) ? 0.0f : result.x;
    result.y = isnan(result.y) ? 0.0f : result.y;
    result.z = isnan(result.z) ? 0.0f : result.z;

    float3 accum = result / (float)spl;
    albedo = albedo / (float)spl;
    normal = normal / (float)spl;

    if (frame > 0)
    {
        const float a = 1.0f / (float)(frame + 1);
        const float3 accum_prev = make_float3(params.accum_buffer[image_idx]);
        accum = lerp(accum_prev, accum, a);
        const float3 albedo_prev = make_float3(params.albedo_buffer[image_idx]);
        const float3 normal_prev = make_float3(params.normal_buffer[image_idx]);
        albedo = lerp(albedo_prev, albedo / (float)spl, a);
        normal = lerp(normal_prev, normal / (float)spl, a);
    }
    params.accum_buffer[image_idx] = make_float4(accum, 1.0f);
    uchar3 color = make_color(accum);
    params.result_buffer[image_idx] = make_float4(color2float(color), 1.0f);
    params.normal_buffer[image_idx] = make_float4(normal, 1.0f);
    params.albedo_buffer[image_idx] = make_float4(albedo, 1.0f);
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

    float3 p = normalize(ray.at(t));

    float phi = atan2(p.z, p.x);
    float theta = asin(p.y);
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    si->uv = make_float2(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<float3, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data
        );
}

// --------------------------------------------------------------------------
// Callables for surface
// --------------------------------------------------------------------------
// Diffuse -----------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_diffuse(SurfaceInteraction* si, void* mat_data) {
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(mat_data);

    if (diffuse->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wi, si->shading.n);

    si->trace_terminate = false;
    uint32_t seed = si->seed;
    const float z0 = rnd(seed);
    const float z1 = rnd(seed);
    float3 wi = cosineSampleHemisphere(z0, z1);
    Onb onb(si->shading.n);
    onb.inverseTransform(wi);
    si->wo = normalize(wi);
    si->seed = seed;
}

extern "C" __device__ float3 __continuation_callable__bsdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(mat_data);
    const float4 albedo = optixDirectCall<float4, SurfaceInteraction*, void*>(diffuse->tex_program_id, si, diffuse->tex_data);
    si->albedo = make_float3(albedo) * albedo.w;
    si->emission = make_float3(0.0f);
    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wo));
    return si->albedo * (cosine / math::pi);
}

extern "C" __device__ float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wo));
    return cosine / math::pi;
}

// Dielectric --------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_dielectric(SurfaceInteraction* si, void* mat_data) {
    const DielectricData* dielectric = reinterpret_cast<DielectricData*>(mat_data);

    float ni = 1.0f; // air
    float nt = dielectric->ior;  // ior specified 
    float cosine = dot(si->wi, si->shading.n);
    bool into = cosine < 0;
    float3 outward_normal = into ? si->shading.n : -si->shading.n;

    if (!into) swap(ni, nt);

    cosine = fabs(cosine);
    float sine = sqrtf(1.0 - cosine*cosine);
    bool cannot_refract = (ni / nt) * sine > 1.0f;

    float reflect_prob = fresnel(cosine, ni, nt);
    unsigned int seed = si->seed;

    if (cannot_refract || reflect_prob > rnd(seed))
        si->wo = reflect(si->wi, outward_normal);
    else    
        si->wo = refract(si->wi, outward_normal, cosine, ni, nt);
    si->radiance_evaled = false;
    si->trace_terminate = false;
    si->seed = seed;
}

extern "C" __device__ float3 __continuation_callable__bsdf_dielectric(SurfaceInteraction* si, void* mat_data)
{
    const DielectricData* dielectric = reinterpret_cast<DielectricData*>(mat_data);
    si->emission = make_float3(0.0f);
    float4 albedo = optixDirectCall<float4, SurfaceInteraction*, void*>(dielectric->tex_program_id, si, dielectric->tex_data);
    si->albedo = make_float3(albedo);
    return si->albedo;
}

extern "C" __device__ float __direct_callable__pdf_dielectric(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}

// Conductor --------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_conductor(SurfaceInteraction* si, void* mat_data) {
    const ConductorData* conductor = reinterpret_cast<ConductorData*>(mat_data);
    if (conductor->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wi, si->shading.n);

    si->wo = reflect(si->wi, si->shading.n);
    si->trace_terminate = false;
    si->radiance_evaled = false;
}

extern "C" __device__ float3 __continuation_callable__bsdf_conductor(SurfaceInteraction* si, void* mat_data)
{
    const ConductorData* conductor = reinterpret_cast<ConductorData*>(mat_data);
    si->emission = make_float3(0.0f);
    float4 albedo = optixDirectCall<float4, SurfaceInteraction*, void*>(conductor->tex_program_id, si, conductor->tex_data);
    si->albedo = make_float3(albedo);
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
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    Onb onb(si->shading.n);

    if (rnd(seed) < diffuse_ratio)
    {
        float3 w_in = cosineSampleHemisphere(z1, z2);
        onb.inverseTransform(w_in);
        si->wo = normalize(w_in);
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
        si->wo = normalize(reflect(si->wi, h));
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
extern "C" __device__ float3 __continuation_callable__bsdf_disney(SurfaceInteraction* si, void* mat_data)
{   
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);
    si->emission = make_float3(0.0f);

    const float3 V = -normalize(si->wi);
    const float3 L = normalize(si->wo);
    const float3 N = normalize(si->shading.n);

    const float NdotV = fabs(dot(N, V));
    const float NdotL = fabs(dot(N, L));

    if (NdotV == 0.0f || NdotL == 0.0f)
        return make_float3(0.0f);

    const float3 H = normalize(V + L);
    const float NdotH = dot(N, H);
    const float LdotH /* = VdotH */ = dot(L, H);

    const Spectrum base_color = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(
        disney->base.prg_id, si, disney->base.data
    );
    si->albedo = base_color;

    // Diffuse term (diffuse, subsurface, sheen) ======================
    // Diffuse
    const float Fd90 = 0.5f + 2.0f * disney->roughness * LdotH*LdotH;
    const float FVd90 = fresnelSchlickT(NdotV, Fd90);
    const float FLd90 = fresnelSchlickT(NdotL, Fd90);
    const float3 f_diffuse = (base_color / math::pi) * FVd90 * FLd90;

    // Subsurface
    const float Fss90 = disney->roughness * LdotH*LdotH;
    const float FVss90 = fresnelSchlickT(NdotV, Fss90);
    const float FLss90 = fresnelSchlickT(NdotL, Fss90); 
    const float3 f_subsurface = (base_color / math::pi) * 1.25f * (FVss90 * FLss90 * ((1.0f / (NdotV * NdotL)) - 0.5f) + 0.5f);

    // Sheen
    const float3 rho_tint = base_color / luminance(base_color);
    const float3 rho_sheen = lerp(make_float3(1.0f), rho_tint, disney->sheen_tint);
    const float3 f_sheen = disney->sheen * rho_sheen * powf(1.0f - LdotH, 5.0f);

    // Specular term (specular, clearcoat) ============================
    // Spcular
    const float3 X = si->shading.dpdu;
    const float3 Y = si->shading.dpdv;
    const float alpha = fmaxf(0.001f, disney->roughness);
    const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
    const float ax = fmaxf(0.001f, math::sqr(alpha) / aspect);
    const float ay = fmaxf(0.001f, math::sqr(alpha) * aspect);
    const float3 rho_specular = lerp(make_float3(1.0f), rho_tint, disney->specular_tint);
    const float3 Fs0 = lerp(0.08f * disney->specular * rho_specular, base_color, disney->metallic);
    const float3 FHs0 = fresnelSchlickR(LdotH, Fs0);
    const float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);
    const float3 f_specular = FHs0 * Ds * Gs;

    // Clearcoat
    const float Fcc = fresnelSchlickR(LdotH, 0.04f);
    const float alpha_cc = 0.1f + (0.001f - 0.1f) * disney->clearcoat_gloss; // lerp
    const float Dcc = GTR1(NdotH, alpha_cc);
    // const float Gcc = smithG_GGX(N, V, L, 0.25f);
    const float Gcc = smithG_GGX(NdotV, 0.25f);
    const float3 f_clearcoat = make_float3( 0.25f * disney->clearcoat * (Fcc * Dcc * Gcc) );

    const float3 out = ( 1.0f - disney->metallic ) * ( lerp( f_diffuse, f_subsurface, disney->subsurface ) + f_sheen ) + f_specular + f_clearcoat;
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

    const float3 V = -si->wo;
    const float3 L = si->wi;
    const float3 N = si->shading.n;

    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    const float specular_ratio = 1.0f - diffuse_ratio;

    const float NdotL = abs(dot(N, L));
    const float NdotV = abs(dot(N, V));

    const float alpha = fmaxf(0.001f, disney->roughness);
    const float alpha_cc = 0.1f + (0.001f - 0.1f) * disney->clearcoat_gloss; // lerp
    const float3 H = normalize(V + L);
    const float NdotH = abs(dot(H, N));

    const float pdf_Ds = GTR2(NdotH, alpha);
    const float pdf_Dcc = GTR1(NdotH, alpha_cc);
    const float ratio = 1.0f / (1.0f + disney->clearcoat);
    const float pdf_specular = (pdf_Dcc + ratio * (pdf_Ds - pdf_Dcc))/* / (4.0f * NdotL * NdotV) */;
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
    si->albedo = make_float3(base);
    
    si->emission = si->albedo * area->intensity * is_emitted;
}

// --------------------------------------------------------------------------
// Callables for texture
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Hitgroups
// --------------------------------------------------------------------------


