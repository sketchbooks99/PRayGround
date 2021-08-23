#include "shape.h"

namespace oprt {

void Shape::attachSurface(const std::shared_ptr<Material>& material)
{
    m_surface = material;
}

void Shape::attachSurface(const std::shared_ptr<AreaEmitter>& area_emitter)
{
    m_surface = area_emitter;
}

void* Shape::surfaceDevicePtr() const
{
    return std::visit([](const auto& surface) { return surface->devicePtr(); }, m_surface);
}

SurfaceType Shape::surfaceType() const
{
    if (std::holds_alternative<std::shared_ptr<Material>>(m_surface))
        return SurfaceType::Material;
    else
        return SurfaceType::AreaEmitter;
}

std::variant<std::shared_ptr<Material>, std::shared_ptr<AreaEmitter>> Shape::surface() const
{
    return m_surface;
}

//void Shape::addProgram(const ProgramGroup& program)
//{
//    if (program.kind() != OPTIX_PROGRAM_GROUP_KIND_HITGROUP)
//    {
//        Message(MSG_ERROR, "oprt::Shape::addProgram(): The kind of input program is not a OPTIX_PROGRAM_GROUP_KIND_HITGROUP.");
//        return;
//    }
//    m_programs.push_back(std::make_unique<ProgramGroup>(program));
//}
//
//std::vector<ProgramGroup> Shape::programs() const
//{
//    std::vector<ProgramGroup> prg_groups;
//    std::transform(m_programs.begin(), m_programs.end(), std::back_inserter(prg_groups), 
//        [](auto prg_ptr) { return *prg_ptr; });
//    return prg_groups;
//}
//
//ProgramGroup Shape::programAt(int idx) const
//{
//    return *m_programs[idx];
//}

void Shape::setSbtIndex(const uint32_t sbt_idx)
{
    m_sbt_index = sbt_idx;
}

uint32_t Shape::sbtIndex() const
{
    return m_sbt_index;
}

void Shape::free()
{
    freeAabbBuffer();
    cuda_free(d_data);
}

void Shape::freeAabbBuffer()
{
    if (d_aabb_buffer) cuda_free(d_aabb_buffer);
}

void* Shape::devicePtr() const
{
    return d_data;
}

} // ::oprt
