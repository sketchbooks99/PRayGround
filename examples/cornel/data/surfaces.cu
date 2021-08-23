#include "util.cuh"
#include <prayground/material/diffuse.h>
#include <prayground/material/dielectric.h>
#include <prayground/emitter/area.h>
#include <prayground/core/bsdf.h>

using namespace prayground;

extern "C" __device__ void __continuation_callable__diffuse(SurfaceInteraction* si, void* surface_data) 
{
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(surface_data);

    if (diffuse->twosided)
        si->n = faceforward(si->n, -si->wi, si->n);

    unsigned int seed = si->seed;
    si->trace_terminate = false;
    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 wi = randomSampleHemisphere(seed);
        Onb onb(si->n);
        onb.inverseTransform(wi);
        si->wo = normalize(wi);
    }
    si->seed = seed;
    si->attenuation *= optixDirectCall<float3, SurfaceInteraction*, void*>(
        diffuse->tex_program_id, si, diffuse->tex_data);
}

extern "C" __device__ void __continuation_callable__dielectric(SurfaceInteraction* si, void* surface_data)
{
    const DielectricData* dielectric = reinterpret_cast<DielectricData*>(surface_data);

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
    si->seed = seed;
    si->radiance_evaled = false;
    si->trace_terminate = false;
    si->attenuation *= optixDirectCall<float3, SurfaceInteraction*, void*>(
        dielectric->tex_program_id, si, dielectric->tex_data);
}

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