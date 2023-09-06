// Opacity Micro Map
// Supported with Ada GPU and after OptiX 7.6

#pragma once

#include <optix.h>
#include <functional>
#include <prayground/optix/context.h>
#include <prayground/optix/macros.h>
#include <prayground/texture/bitmap.h>

namespace prayground {

    class OpacityMicromap {
    public:
        struct MicroBarycentrics {
            Vec2f uv0;
            Vec2f uv1;
            Vec2f uv2;
        };

        using OpacityFunction = std::function<int(const MicroBarycentrics&, const Vec2f& uv0, const Vec2f& uv1, const Vec2f& uv2)>;

        struct Input {
            // Setting for each opacity micro-map
            uint32_t subdivision_level;
            OptixOpacityMicromapFormat format;

            // Texture coordinate and faces of triangle input
            const Vec2f* texcoords;
            const Vec3i* faces;
            uint32_t num_faces;

            // Bitmap texture or lambda function to determine opacity map
            std::variant <std::shared_ptr<BitmapTexture>, std::shared_ptr<FloatBitmapTexture>, OpacityFunction> opacity_bitmap_or_function;
        };

        OpacityMicromap();

        void build(const Context& ctx, CUstream stream, const Input& input, uint32_t build_flags);
        void build(const Context& ctx, CUstream stream, const std::vector<Input>& input, uint32_t build_flags);

        OptixBuildInputOpacityMicromap getBuildInputForGAS() const;

    private:
        template <typename U>
        int evaluateOpacitymapFromBitmap(OptixOpacityMicromapFormat format, const std::shared_ptr<U>& bitmap, const MicroBarycentrics& bc, const Vec2f& uv0, const Vec2f& uv1, const Vec2f& uv2);
        /*void buildFromOpacitymap(const Context& ctx, uint16_t* opacity_map, const Input& input, uint32_t build_flags);
        void constructOpacitymap(uint16_t* out_opacity_map, const Input& input);*/

        OptixMicromapBuffers m_buffers{};
        std::vector<OptixOpacityMicromapUsageCount> m_usage_counts;
    };

} // namespace prayground