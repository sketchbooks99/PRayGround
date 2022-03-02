#pragma once

#include <prayground/math/vec_math.h>
#include <prayground/core/bsdf.h>
#include <prayground/material/conductor.h>
#include <prayground/core/interaction.h>

namespace prayground {

CALLABLE_FUNC void DC_FUNC(sample_conductor)(SurfaceInteraction* si, void* mat_data) {
    const Conductor::Data* conductor = reinterpret_cast<Conductor::Data*>(mat_data);
    if (conductor->twosided) 
        si->n = faceforward(si->n, -si->wi, si->n);

    si->wo = reflect(si->wi, si->n);
    si->trace_terminate = false;
    si->radiance_evaled = false;
}

CALLABLE_FUNC float3 CC_FUNC(bsdf_conductor)(SurfaceInteraction* si, void* mat_data)
{
    const Conductor::Data* conductor = reinterpret_cast<Conductor::Data*>(mat_data);
    si->emission = make_float3(0.0f);
    return optixDirectCall<float3, SurfaceInteraction*, void*>(conductor->tex_program_id, si, conductor->tex_data);
}

CALLABLE_FUNC float DC_FUNC(pdf_conductor)(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}

}