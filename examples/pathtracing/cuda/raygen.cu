#include "util.cuh"
#include <prayground/core/color.h>
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

    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    float3 normal = make_float3(0.0f);
    float p_depth = 0.0f;
    float3 albedo = make_float3(0.0f);

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

            trace(
                params.handle, 
                ro,   
                rd,
                0.01f, 
                tmax, 
                0,      // ray type
                &si
            );

            if (si.trace_terminate) {
                result += si.emission * throughput;
                break;
            }

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

                if (depth == 0) {
                    albedo = si.albedo;
                    float3 op = si.p - ro;
                    float op_length = length(si.p - ro);
                    p_depth = dot(normalize(op), normalize(raygen->camera.lookat - ro)) * op_length;
                    p_depth = p_depth / raygen->camera.farclip;
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
                // BSDFによる重点サンプリング
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id,
                    &si,
                    si.surface_info.data
                    );

                // BSDFのPDFを評価
                float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                    si.surface_info.pdf_id,
                    &si,
                    si.surface_info.data
                    );

                // BSDFの評価
                float3 bsdf_val = optixContinuationCall<float3, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id,
                    &si,
                    si.surface_info.data
                    );

                throughput *= bsdf_val / bsdf_pdf;
            }

            if (depth == 0) {
                albedo += si.albedo;
                float3 op = si.p - ro;
                float op_length = length(si.p - ro);
                p_depth += (dot(normalize(op), normalize(raygen->camera.lookat - ro)) * op_length) / raygen->camera.farclip;
                normal += si.n;
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

    normal = normal / static_cast<float>(params.samples_per_launch);
    p_depth = p_depth / static_cast<float>(params.samples_per_launch);
    albedo = albedo / static_cast<float>(params.samples_per_launch);

    const float3 prev_n = params.normal_buffer[image_index];
    const float prev_d = params.depth_buffer[image_index];
    const float3 prev_a = params.albedo_buffer[image_index];

    if (subframe_index > 0)
    {
        const float a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
        normal = lerp(prev_n, normal, a);
        albedo = lerp(prev_a, albedo, a);
        if (p_depth != 0.0f && prev_d != 0.0f)
            p_depth = prev_d + a * (p_depth - prev_d);
    }
    params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    uchar3 color = make_color(reinhardToneMap(accum_color, params.white));
    params.result_buffer[image_index] = make_uchar4(color.x, color.y, color.z, 255);
    params.normal_buffer[image_index] = normal;
    params.albedo_buffer[image_index] = albedo;
    params.depth_buffer[image_index] = p_depth == 0.0f ? 1.0f : p_depth;
}