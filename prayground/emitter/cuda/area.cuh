#pragma once

#include <prayground/math/vec_math.h>
#include <prayground/core/ray.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/interaction.h>
#include "../area.h"

namespace prayground {

#ifdef __CUDACC__

CALLABLE_FUNC void DC_FUNC(area_emitter)(SurfaceInteraction* si, void* emitterdata)
{
    const AreaEmitterData* area = reinterpret_cast<AreaEmitterData*>(emitterdata);
    si->trace_terminate = true;
    float is_emitted = 1.0f;
    if (!area->twosided)
        is_emitted = dot(si->wi, si->n) < 0.0f ? 1.0f : 0.0f;
    
    si->emission = optixDirectCall<float3, SurfaceInteraction*, void*>(
        area->tex_func_id, si, area->texdata) * area->strength * is_emitted;
}

#endif

}