#pragma once

#include <prayground/core/attribute.h>
#include <prayground/math/vec_math.h>
#include <prayground/shape/trianglemesh.h>
#include <filesystem>
#include <vector>

namespace prayground {

void loadObj(
    const std::filesystem::path& filepath, 
    std::vector<float3>& vertices,
    std::vector<Face>& faces,
    std::vector<float3>& normals,
    std::vector<float2>& texcoords
);

void loadObj(
    const std::filesystem::path& filepath, 
    TriangleMesh& mesh
);

void loadObjWithMtl(
    const std::filesystem::path& objpath, 
    std::vector<float3>& vertices,
    std::vector<Face>& faces,
    std::vector<float3>& normals,
    std::vector<float2>& texcoords, 
    std::vector<uint32_t>& face_indices,
    std::vector<Attributes>& material_attribs,
    const std::filesystem::path& mtlpath
);

void loadObjWithMtl(
    const std::filesystem::path& objpath, 
    const std::filesystem::path& mtlpath, 
    TriangleMesh& mesh, 
    std::vector<Attributes>& material_attribs
);

// If .mtl file exists in same directory of .obj file
void loadObjWithMtl(
    const std::filesystem::path& filepath, 
    TriangleMesh& mesh, 
    std::vector<Attributes>& material_attribs
);

void loadPly(
    const std::filesystem::path& filepath, 
    std::vector<float3>& vertices,
    std::vector<Face>& faces, 
    std::vector<float3>& normals,
    std::vector<float2>& texcoords
);

} // ::prayground