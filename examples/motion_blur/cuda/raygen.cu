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

    float3 result = make_float3(0.0f);

    SurfaceInteraction si;

    unsigned int seed = tea<4>(idx.y * params.width + idx.x, params.subframe_index);

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
        rnd(seed),
        &si
    );

    result = si.shading_val;

    const uint3 launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;

    if (result.x != result.x) result.x = 0.0f;
    if (result.y != result.y) result.y = 0.0f;
    if (result.z != result.z) result.z = 0.0f;
    
    uchar3 color = make_color(result);
    params.result_buffer[image_index] = make_uchar4(color.x, color.y, color.z, 255);
}