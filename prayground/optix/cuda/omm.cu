#include <optix.h>
#include <optix_micromap.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <prayground/math/vec.h>
#include <prayground/optix/util.h>
#include <prayground/optix/omm.h>

namespace prayground {
    INLINE DEVICE float signedTriangleArea(Vec2f a, Vec2f b, Vec2f c) {
        return cross(b - a, c - a) / 2.0f;
    }

    DEVICE uint8_t evaluateTransparencyInSingleMicroTriangle(
        OptixOpacityMicromapFormat format,
        Vec2f uv0, Vec2f uv1, Vec2f uv2,
        OpacityMicromap::MicroBarycentrics bc, 
        Vec2i tex_size,
        cudaTextureObject_t texture)
    {
        Vec2f step = Vec2f(1.0f) / Vec2f(tex_size);

        const Vec2f bary0 = barycentricInterop(uv0, uv1, uv2, bc.uv0);
        const Vec2f bary1 = barycentricInterop(uv0, uv1, uv2, bc.uv1);
        const Vec2f bary2 = barycentricInterop(uv0, uv1, uv2, bc.uv2);

        Vec2f corner_min = Vec2f{
            fminf(fminf(bary0.x(), bary1.x()), bary2.x()), 
            fminf(fminf(bary0.y(), bary1.y()), bary2.y())
        };
        Vec2f corner_max = Vec2f{
            fmaxf(fmaxf(bary0.x(), bary1.x()), bary2.x()), 
            fmaxf(fmaxf(bary0.y(), bary1.y()), bary2.y())
        };

        // Rasterize all pixels in the bounding box of the micro triangle
        int32_t num_pixels_in_triangle = 0;
        int32_t num_transparent_pixels = 0;
        for (float y = corner_min.y(); y <= corner_max.y(); y += step.y()) {
            for (float x = corner_min.x(); x <= corner_max.x(); x += step.x()) {
                Vec2f uv(x, y);
                auto area01 = signedTriangleArea(uv, bary0, bary1);
                auto area12 = signedTriangleArea(uv, bary1, bary2);
                auto area20 = signedTriangleArea(uv, bary2, bary0);
                // Accumulate the number of transparent pixels inside the micro triangle
                if ((area01 >= 0 && area12 >= 0 && area20 >= 0) || (area01 <= 0 && area12 <= 0 && area20 <= 0)) {
                    num_pixels_in_triangle++;
                    float4 color = tex2D<float4>(texture, uv.x(), uv.y());
                    if (color.w <= 0.0f) {
                        num_transparent_pixels++;
                    }
                }
            }
        }

        if (num_transparent_pixels == num_pixels_in_triangle) {
            return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
        } else if (num_transparent_pixels == 0) {
            return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
        } else {
            if (format == OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE)
                return OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
            else
                return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
        }
    }

    extern "C" GLOBAL void generateOpacityMap(
        uint32_t* d_out_omm_data, 
        int32_t subdivision_level,
        int32_t num_faces,
        OptixOpacityMicromapFormat format,
        Vec2i tex_size, 
        const Vec2f* d_texcoords, const Vec3i* d_faces,
        cudaTextureObject_t texture) 
    {
        const int num_micro_triangles = 1 << (subdivision_level * 2);

        int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_thread_id >= num_micro_triangles * num_faces)
            return;

        int face_idx = global_thread_id / num_micro_triangles;
        int micro_tri_idx = global_thread_id % num_micro_triangles;
            
        int num_states_per_elem = 32 / format;

        const Vec2f uv0 = d_texcoords[d_faces[face_idx].x()];
        const Vec2f uv1 = d_texcoords[d_faces[face_idx].y()];
        const Vec2f uv2 = d_texcoords[d_faces[face_idx].z()];

        float2 bary0, bary1, bary2;
        optixMicromapIndexToBaseBarycentrics(micro_tri_idx, subdivision_level, bary0, bary1, bary2);
        
        OpacityMicromap::MicroBarycentrics bc{bary0, bary1, bary2};

        uint8_t state = evaluateTransparencyInSingleMicroTriangle(format, uv0, uv1, uv2, bc, tex_size, texture);
        int index = global_thread_id / num_states_per_elem;
        unsigned int* address = &d_out_omm_data[index];
        atomicOr(address, state << (micro_tri_idx % num_states_per_elem * format));
    }

    extern "C" HOST void evaluateSingleOpacityTexture(
        uint32_t * d_out_omm_data, // GPU pointer to the output opacity map
        int32_t subdivision_level,
        int32_t num_faces,
        OptixOpacityMicromapFormat format,
        Vec2i tex_size,
        const Vec2f * d_texcoords, const Vec3i * d_faces,
        cudaTextureObject_t texture
    ) {
        constexpr int NUM_MAX_THREADS = 1024;

        const int num_micro_triangles = 1 << (subdivision_level * 2);
        const int num_states_per_elem = 32 / format;
        const int total_threads = num_micro_triangles * num_faces;
        const int num_thread = min(total_threads / num_states_per_elem, NUM_MAX_THREADS);
        dim3 threads_per_block(num_thread, 1);
        dim3 block_dim(max(total_threads / num_thread, 1), 1);

        generateOpacityMap<<<block_dim, threads_per_block>>>(d_out_omm_data, subdivision_level, num_faces, format, tex_size, d_texcoords, d_faces, texture);
    }
}