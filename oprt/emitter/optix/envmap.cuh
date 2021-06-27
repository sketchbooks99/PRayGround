#pragma once

#include <sutil/vec_math.h>
#include "../../core/ray.h"
#include "../../optix/sbt.h"

namespace oprt {

struct EnvironmentEmitterData {
    void* texdata;
    unsigned int tex_func_id;
};

#ifdef __CUDACC__

CALLABLE_FUNC void MS_FUNC(envmap)()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    EnvironmentEmitterData* env = reinterpret_cast<EnvironmentEmitterData*>(data->env_data);
    SurfaceInteraction* si = getSurfaceInteraction();

    Ray ray = getWorldRay();
    // float3 p = normalize(ray.at(ray.tmax - length(ray.o)));
    float3 p = normalize(ray.at(1e8f - length(ray.o)));

    float phi = atan2(p.z, p.x);
    float theta = asin(p.y);
    float u = 1.0f - (phi + M_PIf) / (2.0f * M_PIf);
    float v = 1.0f - (theta + M_PIf / 2.0f) / M_PIf;
    si->uv = make_float2(u, v);
    si->trace_terminate = true;
    si->emission = optixDirectCall<float3, SurfaceInteraction*, void*>(
        env->tex_func_id, si, env->texdata
    );
}

#endif

}