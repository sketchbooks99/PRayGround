#include "util.cuh"
#include <oprt/emitter/envmap.h>
#include <oprt/core/ray.h>

using namespace oprt;

extern "C" __device__ void __miss__envmap()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    EnvironmentEmitterData* env = reinterpret_cast<EnvironmentEmitterData*>(data->env_data);
    SurfaceInteraction* si = getSurfaceInteraction();

    Ray ray = getWorldRay();

    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f*1e8f;
    const float discriminant = half_b * half_b - a*c;

    float sqrtd = sqrtf(discriminant);
    float t = (-half_b + sqrtd) / a;

    float3 p = normalize(ray.at(t));

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