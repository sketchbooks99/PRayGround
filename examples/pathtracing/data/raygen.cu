#include "util.cuh"
#include <prayground/core/color.h>
#include "../params.h"

using namespace prayground;

static __forceinline__ __device__ void cameraFrame(const CameraData& camera, float3& U, float3& V, float3& W)
{
    W = camera.lookat - camera.origin;
    float wlen = length(W);
    U = normalize(cross(W, camera.up));
    V = normalize(cross(W, U));

    float vlen = wlen * tanf(0.5f * camera.fov * math::pi / 180.0f);
    V *= vlen;
    float ulen = vlen * camera.aspect;
    U *= ulen;
}

static __forceinline__ __device__ void getCameraRay(const CameraData& camera, const float x, const float y, float3& ro, float3& rd)
{
    float3 U, V, W;
    cameraFrame(camera, U, V, W);

    rd = normalize(x * U + y * V + W);
    ro = camera.origin;
}

/// @todo MISの実装

extern "C" __device__ void __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int subframe_index = params.subframe_index;
    const uint3 idx = optixGetLaunchIndex();

    float3 L = make_float3(0.0f, 0.0f, 0.0f);
    int i = params.samples_per_launch;

    do
    {
        SurfaceInteraction si;
        init_rand_state(&si, make_uint2(params.width, params.height), idx, subframe_index);

        const float2 subpixel_jitter = make_float2(curand_uniform(si.curand_state), curand_uniform(si.curand_state));

        const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(params.width),
            (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(params.height)
        ) - 1.0f;

        float3 ro, rd;
        getCameraRay(raygen->camera, d.x, d.y, ro, rd);

        float3 throughput = make_float3(1.0f);

        si.emission = make_float3(0.0f);
        si.attenuation = make_float3(1.0f);
        si.trace_terminate = false;
        si.radiance_evaled = false;
        si.is_specular     = false;

        int depth = 0;
        for ( ;; ) {

            if ( depth >= params.max_depth )
				break;

            trace(
                params.handle, 
                ro, 
                rd, 
                0.01f, 
                1e16f, 
                &si
            );
            
            if ( si.trace_terminate ) {
                result += si.emission * throughput;
                break;
            }

            if ( si.surface_type == SurfaceType::AreaEmitter )
            {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.brdf_id, 
                    &si, 
                    si.surface_info.data
                );
                result += si.emission * throughput;
                if (si.trace_terminate)
                    break;
            }
            else if ( si.surface_type & SurfaceType::Material )
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
                
                // Evaluate pdf
                float pdf_val = optixDirectCall<float, SurfaceInteraction*, void*>(
                    si.surface_info.pdf_id, 
                    &si,  
                    si.surface_info.data
                );
                
                throughput *= (bsdf_val * pdf_val) / pdf_val;
                result += si.emission * throughput;
            }
            
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
    uchar3 color = make_color(accum_color);
    params.result_buffer[image_index] = make_uchar4(color.x, color.y, color.z, 255);
}