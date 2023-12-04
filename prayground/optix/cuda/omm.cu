#pragma once

#include <optix.h>
#include <optix_micromap.h>
#include <vector_types.h>
#include <prayground/math/vec.h>
#include <prayground/optix/util.h>
#include <prayground/optix/omm.h>

namespace prayground {
    INLINE float signedTriangleArea(Vec2f a, Vec2f b, Vec2f c) {
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

        Vec2f corner_min = Vec2f{fminf(fminf(bary0.x(), bary1.x()), bary2.x()), fminf(fminf(bary0.y(), bary1.y()), bary2.y())};
        Vec2f corner_max = Vec2f{fmaxf(fmaxf(bary0.x(), bary1.x()), bary2.x()), fmaxf(fmaxf(bary0.y(), bary1.y()), bary2.y())};

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
                    if (color.w == 0.0f) {
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

    GLOBAL void generateOpacityMap(
        uint16_t* d_out_omm_data, 
        int32_t subdivision_level, OptixOpacityMicromapFormat format,
        Vec2i tex_size, 
        const Vec2f* d_texcoords, const Vec3i* d_faces,
        cudaTextureObject_t texture) 
    {
        /// TODO: Atomic operation
        const int num_micro_triangles = 1 << (subdivision_level * 2);

        int face_idx = blockIdx.x;
        int micro_tri_idx = threadIdx.x;

        int num_states_per_elem = 16 / format;
        int num_elems_per_face = (num_micro_triangles / 16 * format) + 1;

        const Vec2f uv0 = d_texcoords[d_faces[face_idx].x()];
        const Vec2f uv1 = d_texcoords[d_faces[face_idx].y()];
        const Vec2f uv2 = d_texcoords[d_faces[face_idx].z()];

        float2 bary0, bary1, bary2;
        optixMicromapIndexToBaseBarycentrics(micro_tri_idx, subdivision_level, bary0, bary1, bary2);
        
        OpacityMicromap::MicroBarycentrics bc{bary0, bary1, bary2};

        uint8_t state = evaluateTransparencyInSingleMicroTriangle(format, uv0, uv1, uv2, bc, tex_size, texture);
        int index = face_idx * num_elems_per_face + micro_tri_idx / num_states_per_elem;
        d_out_omm_data[index] |= state << (micro_tri_idx % num_states_per_elem * format);
    }

    extern "C" HOST void generateOpacityMapByTexture(
        uint16_t* d_out_omm_data, // GPU pointer to the output opacity map
        int32_t subdivision_level, 
        int32_t num_faces,
        OptixOpacityMicromapFormat format,
        Vec2i tex_size, 
        const Vec2f* d_texcoords, const Vec3i* d_faces,
        cudaTextureObject_t texture
    ) {
        const int num_micro_triangles = 1 << (subdivision_level * 2);
        dim3 threads_per_block(num_micro_triangles, 1);
        generateOpacityMap<<<num_faces, threads_per_block>>>(d_out_omm_data, tex_size, uv0, uv1, uv2, texture);
    }
}