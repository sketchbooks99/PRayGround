#pragma once

#include <sutil/vec_math.h>
#include <oprt/core/interaction.h>
#include <oprt/core/bsdf.h>
#include <oprt/material/dielectric.h>

CALLABLE_FUNC void DC_FUNC(sample_dielectric)(SurfaceInteraction* si, void* mat_data) {
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
    si->seed = seed;
    si->radiance_evaled = false;
    si->trace_terminate = false;
}

CALLABLE_FUNC float3 CC_FUNC(bsdf_dielectric)(SurfaceInteraction* si, void* mat_data)
{
    const DielectricData* dielectric = reinterpret_cast<DielectricData*>(mat_data);
    si->emission = make_float3(0.0f);
    return optixDirectCall<float3, SurfaceInteraction*, void*>(dielectric->tex_func_id, si, dielectric->texdata);    
}

CALLABLE_FUNC float DC_FUNC(pdf_dielectric)(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}