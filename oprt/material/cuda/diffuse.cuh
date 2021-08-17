#pragma once

#include <sutil/vec_math.h>
#include "../../optix/util.h"
#include "../../core/bsdf.h"
#include "../../core/onb.h"
#include "../diffuse.h"

/**
 * \brief Sample bsdf at the surface.
 * \note This is direct callables function on the device, 
 *       so this function cannot launch `optixTrace()`.  
 */

 CALLABLE_FUNC void DC_FUNC(sample_diffuse)(SurfaceInteraction* si, void* mat_data) {
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(mat_data);

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
}

CALLABLE_FUNC float3 CC_FUNC(bsdf_diffuse)(SurfaceInteraction* si, void* mat_data)
{
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(mat_data);
    const float3 albedo = optixDirectCall<float3, SurfaceInteraction*, void*>(diffuse->tex_func_id, si, diffuse->texdata);
    const float cosine = fmaxf(0.0f, dot(si->n, si->wo));
    si->emission = make_float3(0.0f);
    si->radiance_evaled = false;

    return albedo;
    // return si->n;

    // Next event estimation
    // float3 light_emission = make_float3(0.8f, 0.8f, 0.7f) * 15.0f;
    // unsigned int seed = si->seed;
    // const float z1 = rnd(seed);
    // const float z2 = rnd(seed);
    // si->seed = seed;

    // float3 v1 = make_float3(-130.0f, 0.0f, 0.0f);
    // float3 v2 = make_float3(0.0f, 0.0f, 105.0f);
    // const float3 light_pos = make_float3(343.0f, 548.6f, 227.0f) + v1*z1 + v2*z2;
    
    // const float Ldist = length(light_pos - si->p);
    // const float3 L = normalize(light_pos - si->p);
    // const float nDl = dot(si->n, L);
    // const float LnDl = -dot(make_float3(0.0f, -1.0f, 0.0f), L);
    // float weight = 0.0f;
    // if (nDl > 0.0f && LnDl > 0.0f)
    // {
    //     const bool occluded = traceOcclusion(
    //         params.handle, 
    //         si->p, 
    //         L, 
    //         0.01f, 
    //         Ldist - 0.01f
    //     );

    //     if (!occluded)
    //     {
    //         const float A = length(cross(v1, v2));
    //         weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
    //     }
    // }
    // si->radiance_evaled = true;
    // si->emission = light_emission * make_float3(weight);
    // return albedo * (cosine / M_PIf);
}

CALLABLE_FUNC float DC_FUNC(pdf_diffuse)(SurfaceInteraction* si, void* mat_data)
{
    const float cosine = fmaxf(0.0f, dot(si->n, si->wo));
    return cosine / M_PIf;
}
