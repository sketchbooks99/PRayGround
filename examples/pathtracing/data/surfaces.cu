#include "util.cuh"
#include <prayground/material/diffuse.h>
#include <prayground/material/dielectric.h>
#include <prayground/material/conductor.h>
#include <prayground/material/disney.h>
#include <prayground/emitter/area.h>
#include <prayground/core/bsdf.h>
#include <prayground/core/onb.h>
#include <prayground/core/color.h>

using namespace prayground;

/// @todo 確率密度関数、Next event estimationの実装

// Diffuse -----------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_diffuse(SurfaceInteraction* si, void* mat_data) {
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(mat_data);

    if (diffuse->twosided)
        si->n = faceforward(si->n, -si->wi, si->n);

    si->trace_terminate = false;
    unsigned int seed = si->seed;
    float3 wi = randomSampleHemisphere(seed);
    Onb onb(si->n);
    onb.inverseTransform(wi);
    si->wo = normalize(wi);
    si->seed = seed;
}

extern "C" __device__ float3 __continuation_callable__bsdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(mat_data);
    const float3 albedo = optixDirectCall<float3, SurfaceInteraction*, void*>(diffuse->tex_program_id, si, diffuse->tex_data);
    si->albedo = albedo;
    const float cosine = fmaxf(0.0f, dot(si->n, si->wo));
    return albedo * (cosine / math::pi);
}

extern "C" __device__ float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
    const float cosine = fmaxf(0.0f, dot(si->n, si->wo));
    return cosine / math::pi;
}

// Dielectric --------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_dielectric(SurfaceInteraction* si, void* mat_data) {
    const DielectricData* dielectric = reinterpret_cast<DielectricData*>(mat_data);

    float ni = 1.0f; // air
    float nt = dielectric->ior;  // ior specified 
    float cosine = dot(si->wi, si->n);
    bool into = cosine < 0;
    float3 outward_normal = into ? si->n : -si->n;

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
    float3 albedo = optixDirectCall<float3, SurfaceInteraction*, void*>(dielectric->tex_program_id, si, dielectric->tex_data);
    si->albedo = albedo;
    return albedo;
}

extern "C" __device__ float __direct_callable__pdf_dielectric(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}

// Conductor --------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_conductor(SurfaceInteraction* si, void* mat_data) {
    const ConductorData* conductor = reinterpret_cast<ConductorData*>(mat_data);
    if (conductor->twosided)
        si->n = faceforward(si->n, -si->wi, si->n);

    si->wo = reflect(si->wi, si->n);
    si->trace_terminate = false;
    si->radiance_evaled = false;
}

extern "C" __device__ float3 __continuation_callable__bsdf_conductor(SurfaceInteraction* si, void* mat_data)
{
    const ConductorData* conductor = reinterpret_cast<ConductorData*>(mat_data);
    si->emission = make_float3(0.0f);
    float3 albedo = optixDirectCall<float3, SurfaceInteraction*, void*>(conductor->tex_program_id, si, conductor->tex_data);
    si->albedo = albedo;
    return albedo;
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
        si->n = faceforward(si->n, -si->wi, si->n);

    unsigned int seed = si->seed;
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    Onb onb(si->n);

    if (rnd(seed) < diffuse_ratio)
    {
        float3 w_in = cosineSampleHemisphere(z1, z2);
        onb.inverseTransform(w_in);
        si->wo = normalize(w_in);
    }
    else
    {
        /// @todo Change sampling functions according to the ratio to choose specular or clearcoat pdf
        /// ratio = 1.0f / (1.0f + clearcoat);

        float3 h = sampleGGX(z1, z2, disney->roughness);
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
    const DisneyData* disney = reinterpret_cast<DisneyData*>(mat_data);
    si->emission = make_float3(0.0f);

    const float3 V = -normalize(si->wi);
    const float3 L = normalize(si->wo);
    const float3 N = normalize(si->n);

    const float NdotV = fabs(dot(N, V));
    const float NdotL = fabs(dot(N, L));

    if (NdotV == 0.0f || NdotL == 0.0f)
        return make_float3(0.0f);

    const float3 H = normalize(V + L);
    const float NdotH = dot(N, H);
    const float LdotH /* = VdotH */ = dot(L, H);

    const float3 base_color = optixDirectCall<float3, SurfaceInteraction*, void*>(
        disney->base_program_id, si, disney->base_tex_data
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
    Onb onb(N);
    const float3 X = onb.tangent;
    const float3 Y = onb.bitangent;
    const float alpha = fmaxf(0.001f, disney->roughness * disney->roughness); // Remapping of roughness
    const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
    const float ax = fmaxf(0.001f, math::sqr(disney->roughness) / aspect);
    const float ay = fmaxf(0.001f, math::sqr(disney->roughness) * aspect);
    const float3 rho_specular = lerp(make_float3(1.0f), rho_tint, disney->specular_tint);
    const float3 Fs0 = lerp(0.08f * disney->specular * rho_specular, base_color, disney->metallic);
    const float3 FHs0 = fresnelSchlickR(LdotH, Fs0);
    const float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    const float alpha_g = powf(0.5f*disney->roughness + 0.5f, 2.0f);
    const float Gs = geometrySmith(N, V, L, alpha_g);
    const float3 f_specular = FHs0 * Ds * Gs / (4.0f * NdotV * NdotL);

    // Clearcoat
    const float Fcc = fresnelSchlickR(LdotH, 0.04f);
    const float alpha_cc = 0.1f + (0.001f - 0.1f) * disney->clearcoat_gloss; // lerp
    const float Dcc = GTR1(NdotH, alpha_cc);
    const float Gcc = geometrySmith(N, V, L, 0.25f);
    const float3 f_clearcoat = make_float3( 0.25f * disney->clearcoat * (Fcc * Dcc * Gcc) / (4.0f * NdotV * NdotL) ); 

    const float3 out = ( 1.0f - disney->metallic ) * ( lerp( f_diffuse, f_subsurface, disney->subsurface ) + f_sheen ) + f_specular + f_clearcoat;
    return out * clamp(NdotL, 0.0f, 1.0f) / math::pi;
}

/**
 * @ref http://simon-kallweit.me/rendercompo2015/report/#adaptivesampling
 * 
 * @todo Investigate correct evaluation of PDF.
 */
extern "C" __device__ float __direct_callable__pdf_disney(SurfaceInteraction* si, void* mat_data)
{
    const DisneyData* disney = reinterpret_cast<DisneyData*>(mat_data);

    const float3 V = -si->wi;
    const float3 L = si->wo;
    const float3 N = si->n;

    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    const float specular_ratio = 1.0f - diffuse_ratio;

    const float NdotL = abs(dot(N, L));
    const float NdotV = abs(dot(N, V));

    const float alpha = fmaxf(0.001f, disney->roughness * disney->roughness);
    const float alpha_cc = 0.1f + (0.001f - 0.1f) * disney->clearcoat_gloss; // lerp
    const float3 H = normalize(V + L);
    const float NdotH = abs(dot(H, N));

    const float pdf_Ds = GTR2(NdotH, alpha);
    const float pdf_Dcc = GTR1(NdotH, alpha_cc);
    const float ratio = 1.0f / (1.0f + disney->clearcoat);
    const float pdf_specular = (pdf_Dcc + ratio * (pdf_Ds - pdf_Dcc)) / (4.0f * NdotH);
    const float pdf_diffuse = NdotL / math::pi;

    return diffuse_ratio * pdf_diffuse + specular_ratio * pdf_specular;
}

// Area emitter ------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__area_emitter(SurfaceInteraction* si, void* surface_data)
{
    const AreaEmitterData* area = reinterpret_cast<AreaEmitterData*>(surface_data);
    si->trace_terminate = true;
    float is_emitted = 1.0f;
    if (!area->twosided)
        is_emitted = dot(si->wi, si->n) < 0.0f ? 1.0f : 0.0f;
    
    si->emission = optixDirectCall<float3, SurfaceInteraction*, void*>(
        area->tex_program_id, si, area->tex_data) * area->strength * is_emitted;
}