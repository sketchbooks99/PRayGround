#include "util.cuh"
#include <prayground/emitter/envmap.h>
#include <prayground/core/ray.h>

using namespace prayground;

extern "C" __global__ void __miss__envmap()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    const auto* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
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
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) * math::inv_pi;
    si->shading.uv = Vec2f(u, v);
    si->trace_terminate = true;
    const Vec4f emission = optixDirectCall<Vec4f, const Vec2f&, void*>(
        env->texture.prg_id, si->shading.uv, env->texture.data);
    si->emission = Vec3f(emission);
}

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(1);
}