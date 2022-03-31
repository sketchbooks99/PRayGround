#include "util.cuh"

extern "C" __device__ void __miss__envmap()
{
    const auto* data = (pgMissData*)optixGetSbtDataPointer();
    const auto* env = (EnvironmentEmitter::Data*)data->env_data;

    auto* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    Ray ray = getWorldRay();

    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f * 1e8f;
    const float D = half_b * half_b - a * c;

    const float sqrtD = sqrtf(D);
    const float t = (-half_b + sqrtD) / a;

    const Vec3f p = normalize(ray.at(t));

    const float phi = atan2(p.z(), p.x());
    const float theta = asin(p.y());
    const float u = 1.0f - (phi + math::pi) / (math::two_pi);
    const float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;

    si->shading.uv = Vec2f(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data);
}

extern "C" __device__ void __miss__shadow()
{
    // Shadow ray is not occluded by objects
    setPayload<0>(0);
}