#include "omm.h"

namespace prayground {
    OpacityMicroMap::OpacityMicroMap()
        : m_buffers{{}}
    {
        m_settings.build_flags = OPTIX_OPACITY_MICROMAP_FLAG_NONE;
        m_settings.format = OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE;
        m_settings.subdivision_level = 3; // 4^3 = 64 micro-triangles per single triangle
    }

    OpacityMicroMap::OpacityMicroMap(const Settings& settings)
        : m_settings(settings), m_buffers{{}}
    {

    }

    OpacityMicroMap::OpacityMicroMap(const Settings& settings, const std::shared_ptr<TriangleMesh>& mesh)
        : m_settings{settings}, m_buffers{{}}, m_mesh_ptr{mesh}
    {
    }

    void OpacityMicroMap::setMesh(const std::shared_ptr<TriangleMesh>& mesh)
    {
        m_mesh_ptr = mesh;
    }

    void OpacityMicroMap::build(const Context& ctx, const std::shared_ptr<Texture>& opacity_texture)
    {
        ASSERT(m_mesh_ptr != nullptr, "Mesh to construct OMM hasn't set");
        ASSERT(!m_mesh_ptr->texcoords().empty(), "Texture coordinate buffer to create OMM is empty");
        ASSERT(m_settings.format != OPTIX_OPACITY_MICROMAP_FORMAT_NONE, "Invalid format");

        const int N_TRIANGLES = m_mesh_ptr->faces().size();
        const int N_MICRO_TRIANGLES = 1 << (m_settings.subdivision_level * 2);

        unsigned short** omm_opacity_data = new unsigned short* [N_TRIANGLES];
        for (size_t i = 0; i < N_TRIANGLES; i++)
            omm_opacity_data[i] = new unsigned short[(N_MICRO_TRIANGLES / sizeof(unsigned short)) * m_settings.format];


    }

    void OpacityMicroMap::build(const Context& ctx, const std::function<int(const Vec2f&, const Vec2f&, const Vec2f&, const Vec2f*)> opacity_func)
    {
    }
    OptixBuildInputOpacityMicromap OpacityMicroMap::getBuildInputForGAS() const
    {
        return OptixBuildInputOpacityMicromap();
    }
} // namespace prayground