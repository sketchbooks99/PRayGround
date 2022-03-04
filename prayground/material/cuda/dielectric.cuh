#pragma once

#include <prayground/math/vec_math.h>
#include <prayground/core/interaction.h>
#include <prayground/core/bsdf.h>
#include <prayground/material/dielectric.h>

CALLABLE_FUNC void DC_FUNC(sample_dielectric)(SurfaceInteraction* si, void* mat_data) {
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(mat_data);

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
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(mat_data);
    si->emission = make_float3(0.0f);
    return optixDirectCall<float3, SurfaceInteraction*, void*>(dielectric->tex_program_id, si, dielectric->texture);    
}

CALLABLE_FUNC float DC_FUNC(pdf_dielectric)(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}