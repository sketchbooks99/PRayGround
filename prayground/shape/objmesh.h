#pragma once

#ifndef __CUDACC__
#include <prayground/core/material.h>
#endif
#include <prayground/shape/trianglemesh.h>

namespace prayground {

template <class MaterialT>
class ObjMesh final : public TriangleMesh {
public:
    ObjMesh(const std::filesystem::path& filename);

    constexpr ShapeType type() override;

    OptixBuildInput createBuildInput() override;
private:
    std::vector<std::shared_ptr<MaterialT>> m_materials;
};

} // ::prayground