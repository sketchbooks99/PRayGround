// Displaced Micro Mesh
// Supported after Ada GPU and after 7.7

#pragma once 

#include <optix.h>
#ifndef __CUDACC__
#include <functional>
#include <prayground/optix/macros.h>
#include <prayground/optix/context.h>
#include <prayground/texture/bitmap.h>
#endif

namespace prayground {

    class DisplacedMicromesh {
    public:
        struct Input {
            uint32_t subdivision_level;
            OptixDisplacementMicromapFormat format;

            const Vec2f* texcoords;
            const Vec3i* texcoord_indices;
            const Vec3f* vertices;
            const Vec3i* vertex_indices;
            const uint32_t num_triangles;
        };

        DisplacedMicromesh();

        void build(const Context& ctx, CUstream stream, const Input& input, uint32_t build_flags);

        OptixBuildInputDisplacementMicromap getBuildInputForGAS() const;
    private:
        OptixMicromapBuffers m_buffers{};

    };

} // namespace prayground