#pragma once

#include <sutil/vec_math.h>
#include "../../optix/util.h"

namespace oprt {

struct ConductorData {
    void* texdata;
    float fuzz;
    bool twosided;
    unsigned int tex_func_id;
};

CALLABLE_FUNC void DC_FUNC(sample_conductor)(SurfaceInteraction* si, void* mat_data) {
    const ConductorData* conductor = reinterpret_cast<ConductorData*>(mat_data);
    if (conductor->twosided) 
        si->n = faceforward(si->n, -si->wi, si->n);

    si->wo = reflect(si->wi, si->n);
    si->trace_terminate = false;
    si->radiance_evaled = false;
}

CALLABLE_FUNC float3 CC_FUNC(bsdf_conductor)(SurfaceInteraction* si, void* mat_data)
{
    const ConductorData* conductor = reinterpret_cast<ConductorData*>(mat_data);
    si->emission = make_float3(0.0f);
    return optixDirectCall<float3, SurfaceInteraction*, void*>(conductor->tex_func_id, si, conductor->texdata);
}

CALLABLE_FUNC float DC_FUNC(pdf_conductor)(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}

}