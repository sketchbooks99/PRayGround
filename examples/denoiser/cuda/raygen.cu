#include <prayground/core/color.h>
#include <prayground/core/bsdf.h>
#include "../params.h"
#include "util.cuh"

using namespace prayground;

static __forceinline__ __device__ void getCameraRay(const CameraData& camera, float x, float y, float3& ro, float3& rd)
{
    rd = normalize(x * camera.U + y * camera.V + camera.W);
    ro = camera.origin;
}

static __forceinline__ __device__ float3 reinhardToneMap(const float3& color, const float white)
{
    const float l = luminance(color);
    return (color * 1.0f) / (1.0f + l / white);
}

static __forceinline__ __device__ float3 exposureToneMap(const float3& color, const float exposure)
{
    return make_float3(1.0f) - expf(-color * exposure);
}

extern "C" __device__ void __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int frame = params.frame;
    const uint3 idx = optixGetLaunchIndex();
    unsigned int seed = tea<4>(idx.x * parmas.width + idx.y, frame);

    float3 result = make_float3(0.0f);
    float3 normal = make_float3(0.0f);
    float3 albedo = make_float3(0.0f);

    int spl = params.samples_per_launch;

    for (int i = 0; i < spl; i++)
    {
        const float2 jitter = make_float2(random(seed) - 0.5f, random(seed) - 0.5f);

        const float2 d = 2.0f * make_float2(
            ((float)idx.x + jitter.x) / (float)params.width, 
            (float)idx.y + jitter.y) / (float)params.height
        ) - 1.0f;

        float3 ro, rd;
        getCameraRay(raygen->camera, d.x, d.y, ro, rd);

        float3 throughput = make_float3(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = make_float3(0.0f);
        si.albedo = make_float3(0.0f);
        si.n = make_float3(0.0f);
        si.trace_terminate = false;
        si.radiance_evaled = false;

        float tmax = raygen->camera.farclip / dot(rd, normalize(raygen->camera.W));

        int depth = 0;

        while (true)
        {
            if (depth >= params.max_depth)
                break;

            trace(params.handle, ro, rd, 0.01f, tmax, /* ray_type = */ 0, &si);

            if (si.trace_terminate)
            {
                result += throughput * si.emission;
                break;
            }

            // Get emission from area emitter
            if (si.surface_info.type == SurfaceType::AreaEmitter)
            {
                // Evaluate emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id, 
                    &si,
                    si.surface_info.data
                );
                result += throughput * si.emission;
                
                if (depth == 0)
                {
                    albedo = si.albedo;
                    normal = si.n;
                }

                if (si.trace_terminate)
                    break;
            }
            // Specular sampling
            else if (+(si.surface_info.type & SurfaceType::Delta))
            {
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id,
                    &si,
                    si.surface_info.data
                );

                // Evaluate BSDF
                const float3 bsdf_val = optixContinuationCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id,
                    &si,
                    si.surface_info.data
                );
                throughput *= bsdf_val;
            }
            // Rough surface sampling
            else if (+(si.surface_info.type & (SurfaceType::Rough | SurfaceType::Diffuse)))
            {
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id,
                    &si,
                    si.surface_info.data
                );

                // Evaluate BSDF
                const float3 bsdf_val = optixContinuationCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id,
                    &si,
                    si.surface_info.data
                );

                // Evaluate PDF
                float pdf = optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.pdf_id,
                    &si,
                    si.surface_info.data
                );

                pdf = fmaxf(pdf, math::eps);

                throughput *= bsdf_val * pdf;
            }

            if (depth == 0)
            {
                albedo = si.albedo;
                normal = si.n;
            }

            tmax = 1e16f;

            ro = si.p;
            rd = si.wo;

            depth++;
        }
    }

    const unsigned int image_idx = launch_index.y * params.width + launch_index.x;

    result.x = isnan(result.x) ? 0.0f : result.x;
    result.y = isnan(result.y) ? 0.0f : result.y;
    result.z = isnan(result.z) ? 0.0f : result.z;

    float3 accum = result / (float)spl;

    if (frame > 0)
    {
        const float a = 1.0f / (float)(frame + 1);
        const float3 accum_prev = make_float3(params.accum_buffer[image_idx]);
        accum = lerp(accum_prev, accum, a);
    }
    params.accum_buffer[image_idx] = make_float4(accum, 1.0f);
    params.result_buffer[image_idx] = make_float4(reinhardToneMap(accum, params.white), 1.0f);
    params.normal_buffer[image_idx] = make_float4(normal, 1.0f);
    parmas.albedo_buffer[image_idx] = make_float4(albedo, 1.0f);
}