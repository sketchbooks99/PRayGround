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

        // Check transparency on the micro-triangle vertices
        num_pixels_in_triangle += 3;
        float4 c0 = tex2D<float4>(texture, bary0.x(), bary0.y());
        float4 c1 = tex2D<float4>(texture, bary1.x(), bary1.y());
        float4 c2 = tex2D<float4>(texture, bary2.x(), bary2.y());
        num_transparent_pixels += (int)(c0.w == 0.0f);
        num_transparent_pixels += (int)(c1.w == 0.0f);
        num_transparent_pixels += (int)(c2.w == 0.0f);

        // Shift corner if the pixel step is larger than micro-triangle size
        auto corner_size = corner_max - corner_min;
        if (step.x() > corner_size.x())
            corner_min.x() = (bary0.x() + bary1.x() + bary2.x()) / 3.0f;
        if (step.y() > corner_size.y())
            corner_min.y() = (bary0.y() + bary1.y() + bary2.y()) / 3.0f;

        for (float y = corner_min.y(); y <= corner_max.y(); y += step.y()) {
            for (float x = corner_min.x(); x <= corner_max.x(); x += step.x()) {
                Vec2f uv(x, y);
                auto area01 = signedTriangleArea(uv, bary0, bary1);
                auto area12 = signedTriangleArea(uv, bary1, bary2);
                auto area20 = signedTriangleArea(uv, bary2, bary0);
                // Accumulate the number of transparent pixels inside the micro triangle
                if ((area01 >= 0 && area12 >= 0 && area20 >= 0) || (area01 <= 0 && area12 <= 0 && area20 <= 0)) {
                    num_pixels_in_triangle++;
                    float4 color = tex2D<float4>(texture, x, y);
                    num_transparent_pixels += (int)(color.w == 0.0f);
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
        size_t num_elems_per_face = max((num_micro_triangles / 32 * format), 1);

        //int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        int64_t x_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int64_t y_idx = blockIdx.y * blockDim.y + threadIdx.y;
        int64_t global_thread_id = x_idx + (gridDim.x * gridDim.y * y_idx);
        if (global_thread_id >= num_micro_triangles * num_faces)
            return;

        int64_t face_idx = global_thread_id / num_micro_triangles;
        int64_t micro_tri_idx = global_thread_id % num_micro_triangles;
            
        int num_states_per_elem = 32 / format;
        
        const Vec2f uv0 = d_texcoords[d_faces[face_idx].x()];
        const Vec2f uv1 = d_texcoords[d_faces[face_idx].y()];
        const Vec2f uv2 = d_texcoords[d_faces[face_idx].z()];

        float2 bary0, bary1, bary2;
        optixMicromapIndexToBaseBarycentrics(micro_tri_idx, subdivision_level, bary0, bary1, bary2);
        
        OpacityMicromap::MicroBarycentrics bc{bary0, bary1, bary2};

        uint8_t state = evaluateTransparencyInSingleMicroTriangle(format, uv0, uv1, uv2, bc, tex_size, texture);
        int64_t index = global_thread_id / num_states_per_elem;
        uint32_t* address = &d_out_omm_data[index];
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
        constexpr int NUM_MAX_BLOCK = 65536;

        const int num_micro_triangles = 1 << (subdivision_level * 2);
        // Count the number of opacity states packed in single uint32_t element
        const int num_thread = min(num_micro_triangles, NUM_MAX_THREADS);
        dim3 threads_per_block(num_thread, 1);
        
        const int total_blocks = (num_micro_triangles / num_thread + 1) * num_faces;
        int block_size_x = min(total_blocks, NUM_MAX_BLOCK);
        int block_size_y = max(total_blocks / NUM_MAX_BLOCK, 1);
        dim3 block_dim(block_size_x, block_size_y, 1);

        printf("NumFaces: %d NumMicroTriangles: %d, BlockSize: %d %d %d, GridSize: %d %d %d\n", num_faces, num_micro_triangles, threads_per_block.x, threads_per_block.y, threads_per_block.z, block_dim.x, block_dim.y, block_dim.z);

        generateOpacityMap<<<block_dim, threads_per_block>>>(d_out_omm_data, subdivision_level, num_faces, format, tex_size, d_texcoords, d_faces, texture);
    }
}