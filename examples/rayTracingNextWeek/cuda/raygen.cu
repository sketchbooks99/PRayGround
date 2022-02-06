#include "util.cuh"
#include <prayground/core/spectrum.h>
#include <prayground/core/bsdf.h>
#include "../params.h"

using namespace prayground;

static __forceinline__ __device__ void getCameraRay(const CameraData& camera, const float x, const float y, float3& ro, float3& rd)
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

/// @todo MISの実装

extern "C" __device__ void __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int subframe_index = params.subframe_index;
    const uint3 idx = optixGetLaunchIndex();
    unsigned seed = tea<4>(idx.x * params.width + idx.y, subframe_index);

    float3 result = make_float3(0.0f);
    float3 normal = make_float3(0.0f);

    int i = params.samples_per_launch;

    do
    {
        const float2 subpixel_jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

        const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(params.width),
            (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(params.height)
        ) - 1.0f;

        float3 ro, rd;
        getCameraRay(raygen->camera, d.x, d.y, ro, rd);

        float3 throughput = make_float3(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = make_float3(0.0f);
        si.albedo = make_float3(0.0f);
        si.trace_terminate = false;
        si.radiance_evaled = false;

        float tmax = raygen->camera.farclip / dot(rd, normalize(raygen->camera.lookat - ro));

        int depth = 0;
        for ( ;; ) {

            if ( depth >= params.max_depth )
				break;

            trace(params.handle, ro, rd, 0.01f, tmax, rnd(seed), /* ray_type = */ 0, &si);

            if (si.trace_terminate) {
                result += si.emission * throughput;
                break;
            }

            if (depth == 0)
                normal = si.n;

            // Get emission from area emitter
            if ( si.surface_info.type == SurfaceType::AreaEmitter )
            {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id, 
                    &si, 
                    si.surface_info.data
                );
                result += si.emission * throughput;
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
                
                // Evaluate bsdf
                float3 bsdf_val = optixContinuationCall<float3, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id,
                    &si, 
                    si.surface_info.data
                );
                throughput *= bsdf_val;
            }
            // Rough surface sampling with applying MIS
            else if ( +(si.surface_info.type & (SurfaceType::Rough | SurfaceType::Diffuse)) )
            {
                unsigned int seed = si.seed;
                const int light_id = rnd_int(seed, 0, params.num_lights-1);
                const AreaEmitterInfo light = params.lights[light_id];

                const float weight = 1.0f / (params.num_lights + 1);

                float pdf_val = 0.0f;

                // Importance sampling according to the BSDF
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id,
                    &si,
                    si.surface_info.data
                    );

                if (rnd(seed) < weight * params.num_lights) {
                    // Light sampling
                    float3 to_light = optixDirectCall<float3, AreaEmitterInfo, SurfaceInteraction*>(
                        light.sample_id,
                        light,
                        &si
                        );
                    si.wo = normalize(to_light);
                }

                for (int i = 0; i < params.num_lights; i++)
                {
                    // Evaluate PDF of area emitter
                    float light_pdf = optixContinuationCall<float, AreaEmitterInfo, const float3&, const float3&>(
                        params.lights[i].pdf_id,
                        params.lights[i],
                        si.p,
                        si.wo
                    );
                    pdf_val += weight * light_pdf;
                }

                // Evaluate PDF of area emitter
                float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                    si.surface_info.pdf_id,
                    &si,
                    si.surface_info.data
                );

                pdf_val += weight * bsdf_pdf;

                // Evaluate BSDF
                float3 bsdf_val = optixContinuationCall<float3, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id,
                    &si,
                    si.surface_info.data
                    );

                if (pdf_val == 0.0f) break;

                throughput *= bsdf_val / pdf_val;
            }

            // プライマリーレイ以外ではtmaxは大きくしておく
            tmax = 1e16f;
            
            ro = si.p;
            rd = si.wo;

            ++depth;
        }
    } while (--i);

    const uint3 launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;

    if (result.x != result.x) result.x = 0.0f;
    if (result.y != result.y) result.y = 0.0f;
    if (result.z != result.z) result.z = 0.0f;

    float3 accum_color = result / static_cast<float>(params.samples_per_launch);

    if (subframe_index > 0)
    {
        const float a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    uchar3 color = make_color(reinhardToneMap(accum_color, params.white));
    params.result_buffer[image_index] = make_uchar4(color.x, color.y, color.z, 255);
}