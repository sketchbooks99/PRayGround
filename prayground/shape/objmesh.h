#pragma once

#ifndef __CUDACC__
#include <prayground/core/material.h>
#endif
#include <prayground/shape/trianglemesh.h>

namespace prayground {

    class ObjMesh final : public TriangleMesh {
    public:
        using Data = TriangleMesh::Data;

        ObjMesh(const std::filesystem::path& filename);

        const std::vector<std::shared_ptr<Material>>& materials() const;
    private:
        std::vector<int32_t> m_sbt_indices;
        std::unordered_map<std::string, std::shared_ptr<>>
    };

} // namespace prayground