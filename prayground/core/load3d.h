#pragma once

#include <prayground/core/attribute.h>
#include <prayground/math/vec_math.h>
#include <prayground/shape/trianglemesh.h>
#include <filesystem>
#include <vector>

namespace prayground {

void loadObjWithMtl(
    const std::filesystem::objpath, 
    const std::filesystem::mtlpath, 
    TriangleMesh& mesh, 
    std::vector<Attributes>& material_attribs
);

void loadObj(
    const std::filesystem::path& filepath, 
    std::vector<float3>& vertices,
    std::vector<Face>& faces,
    std::vector<float3>& normals,
    std::vector<float2>& texcoords
);

void loadObj(const std::filesystem::path& filepath, TriangleMesh& mesh);

void loadPly(
    const std::filesystem::path& filepath, 
    std::vector<float3>& vertices,
    std::vector<Face>& faces, 
    std::vector<float3>& normals,
    std::vector<float2>& texcoords
);

} // ::prayground