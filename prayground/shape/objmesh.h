#pragma once

#ifndef __CUDACC__
#include <prayground/core/material.h>
#endif
#include <prayground/shape/trianglemesh.h>

namespace prayground {

template <class MaterialT>
class ObjMesh final : public TriangleMesh {
public:
    using DataType = MeshData;

    ObjMesh(const std::filesystem::path& filename);

    constexpr ShapeType type() override;
private:
    std::vector<std::shared_ptr<MaterialT>> m_materials;
};

} // ::prayground