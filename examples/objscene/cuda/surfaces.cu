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
    uint32_t seed = si->seed;
    const float z0 = rnd(seed);
    const float z1 = rnd(seed);
    float3 wi = cosineSampleHemisphere(z0, z1);
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
    si->emission = make_float3(0.0f);
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

// Area emitter ------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__area_emitter(SurfaceInteraction* si, void* surface_data)
{
    const AreaEmitterData* area = reinterpret_cast<AreaEmitterData*>(surface_data);
    si->trace_terminate = true;
    float is_emitted = dot(si->wi, si->n) < 0.0f ? 1.0f : 0.0f;
    if (area->twosided)
    {
        is_emitted = 1.0f;
        si->n = faceforward(si->n, -si->wi, si->n);
    }

    const float3 base = optixDirectCall<float3, SurfaceInteraction*, void*>(
        area->tex_program_id, si, area->tex_data);
    si->albedo = base;
    
    si->emission = base * area->intensity * is_emitted;
}
