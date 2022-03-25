#include "util.cuh"
#include <prayground/core/spectrum.h>
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

    unsigned int seed = tea<4>(idx.y * params.width + idx.x, params.subframe_index);

    for (int i = 0; i < params.samples_per_launch; i++)
    {
        SurfaceInteraction si;
        const float2 subpixel_jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

        const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(params.width),
            (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(params.height)
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

        result += si.shading_val;
    }

    const uint3 launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;

    if (result.x != result.x) result.x = 0.0f;
    if (result.y != result.y) result.y = 0.0f;
    if (result.z != result.z) result.z = 0.0f;

    float3 accum_color = result / static_cast<float>(params.samples_per_launch);
    if (params.subframe_index > 0)
    {
        const float a = 1.0f / static_cast<float>(params.subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    
    uchar3 color = make_color(accum_color);
    params.result_buffer[image_index] = make_uchar4(color.x, color.y, color.z, 255);
    params.accum_buffer[image_index] = make_float4(accum_color);
}