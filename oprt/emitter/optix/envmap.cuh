#pragma once

#include "envmap.h"

namespace oprt {

CALLABLE_FUNC void MS_FUNC(envmap)()
{
    EnvironmentEmitterData* env = reinterpret_cast<EnvironmentEmitterData*>(optixGetSbtDataPointer());
    SurfaceInteraction* si = getSurfaceInteraction();

    Ray ray = getWorldRay();
    float3 p = normalize(ray.at(ray.tmax - length(ray.o)));

    float phi = atan2(p.z, p.x);
    float theta = asin(p.y);
    float u = 1.0f - (phi + M_PIf) / (2.0f * M_PIf);
    float v = 1.0f - (theta + M_PIf / 2.0f) / M_PIf;
    si->emission = optixDirectCall<float3, SurfaceInteraction*, void*>(env->tex_func_id, si, env->texdata);

    si->trace_terminate = true;
}


}