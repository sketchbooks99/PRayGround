// Opacity Micro Map
// Supported with Ada GPU and after OptiX 7.6

#pragma once

#include <optix.h>
#include <functional>
#include <prayground/optix/context.h>
#include <prayground/shape/trianglemesh.h>
#include <prayground/core/texture.h>

namespace prayground {

    class OpacityMicroMap {
        static_assert(OPTIX_VERSION >= 70600, "Opacity micro map requires OptiX 7.6 at the minimum");

    public:
        struct Settings {
            int32_t subdivision_level;
            OptixOpacityMicromapFormat format;
            uint32_t build_flags;
        };

        OpacityMicroMap();
        OpacityMicroMap(const Settings& settings);
        OpacityMicroMap(const Settings& settings, const std::shared_ptr<TriangleMesh>& mesh);

        void setMesh(const std::shared_ptr<TriangleMesh>& mesh);

        // Build a map with opacity texture. The function will launch CUDA kernel to create opacity map
        // Only the pixel where alpha = 0 will be treated as TRANSPARENT, so other pixels will be OPAQUE or UNKNOWN_OPAQUE
        void build(const Context& ctx, const std::shared_ptr<Texture>& opacity_texture);
        // Build a map with user-defined function to determine opacity in a triangle
        void build(const Context& ctx, const std::function<int(const Vec2f&, const Vec2f&, const Vec2f&, const Vec2f*)> opacity_func);

        OptixBuildInputOpacityMicromap getBuildInputForGAS() const;

    private:
        Settings m_settings;
        OptixMicromapBuffers m_buffers;
        std::shared_ptr<TriangleMesh> m_mesh_ptr;
    };
    
    extern "C" HOST void calculateOpacityMapOnDevice(
        unsigned short** out_opacity_data, 
        const Vec2f* texcoords, const Face* faces, 
        const void* texture_data, Texture::Type texture_type,
        const OpacityMicroMap::Settings& settings);

} // namespace prayground