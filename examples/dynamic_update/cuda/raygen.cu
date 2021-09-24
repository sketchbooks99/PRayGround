#include "util.cuh"
#include <prayground/core/color.h>
#include "../params.h"

using namespace prayground;

static __forceinline__ __device__ void getCameraRay(const CameraData& camera, const float x, const float y, float3& ro, float3& rd)
{
    rd = normalize(x * camera.U + y * camera.V + camera.W);
    ro = camera.origin;
}

extern "C" __device__ void __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const uint3 idx = optixGetLaunchIndex();

    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 normal = make_float3(0.0f);
    float3 albedo = make_float3(0.0f);

    SurfaceInteraction si;

    const float2 d = 2.0f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(params.width),
        static_cast<float>(idx.y) / static_cast<float>(params.height)
    ) - 1.0f;
    float3 ro, rd;
    getCameraRay(raygen->camera, d.x, d.y, ro, rd);

    trace(
        params.handle,
        ro, 
        rd, 
        0.01f, 
        1e16f, 
        &si
    );

    normal = si.n;
    color = si.shading_val;
    albedo = si.albedo;

    const uint3 launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;

    if (color.x != color.x) color.x = 0.0f;
    if (color.y != color.y) color.y = 0.0f;
    if (color.z != color.z) color.z = 0.0f;
    
    uchar3 result = make_color(color);
    params.result_buffer[image_index] = make_uchar4(result.x, result.y, result.z, 255);
    params.normal_buffer[image_index] = normal;
    params.albedo_buffer[image_index] = albedo;
}