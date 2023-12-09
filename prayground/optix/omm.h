// Opacity Micro Map
// Supported with Ada GPU and after OptiX 7.6

#pragma once

#include <optix.h>
#ifndef __CUDACC__
#include <functional>
#include <prayground/optix/context.h>
#include <prayground/optix/macros.h>
#include <prayground/texture/bitmap.h>
#endif

namespace prayground {

    class OpacityMicromap {
    public:
        struct MicroBarycentrics {
            Vec2f uv0;
            Vec2f uv1;
            Vec2f uv2;
        };

#ifndef __CUDACC__
        using OpacityFunction = std::function<int(const MicroBarycentrics&, const Vec2f& uv0, const Vec2f& uv1, const Vec2f& uv2)>;

        struct Input {
            // Setting for each opacity micro-map
            uint32_t subdivision_level;
            OptixOpacityMicromapFormat format;

            // Texture coordinate and faces of triangle input
            const uint32_t num_texcoords;
            const Vec2f* texcoords;
            const uint32_t num_faces;
            const Vec3i* faces;

            // Bitmap texture or lambda function to determine opacity map
            std::variant <std::shared_ptr<BitmapTexture>, std::shared_ptr<FloatBitmapTexture>, OpacityFunction> opacity_bitmap_or_function;
        };

        OpacityMicromap();

        void build(const Context& ctx, CUstream stream, const Input& input, uint32_t build_flags);
        void build(const Context& ctx, CUstream stream, const std::vector<Input>& input, uint32_t build_flags);

        OptixBuildInputOpacityMicromap getBuildInputForGAS() const;

        static MicroBarycentrics indexToBarycentrics(uint32_t index, uint32_t subdivision_level);

    private:
        template <typename U>
        int evaluateOpacitymapFromBitmap(OptixOpacityMicromapFormat format, const std::shared_ptr<U>& bitmap, const MicroBarycentrics& bc, const Vec2f& uv0, const Vec2f& uv1, const Vec2f& uv2);

        OptixMicromapBuffers m_buffers{};
        std::vector<OptixOpacityMicromapUsageCount> m_usage_counts;
#endif
    };

    extern "C" HOST void evaluateSingleOpacityTexture(
        uint16_t* d_out_omm_data, // GPU pointer to the output opacity map
        int32_t subdivision_level,
        int32_t num_faces,
        OptixOpacityMicromapFormat format,
        Vec2i tex_size,
        const Vec2f * d_texcoords, const Vec3i * d_faces,
        cudaTextureObject_t texture
    );

} // namespace prayground