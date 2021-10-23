#pragma once

#ifndef __CUDACC__
#include <prayground/core/material.h>
#endif
#include <prayground/shape/trianglemesh.h>

namespace prayground {

class ObjMesh final : public TriangleMesh {
public:
    using DataType = MeshData;

    ObjMesh(const std::filesystem::path& filename);

    constexpr ShapeType type() override;
private:
    std::vector<int32_t> m_sbt_indices;
};

} // ::prayground