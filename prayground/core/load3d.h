#pragma once

#include <prayground/math/vec_math.h>
#include <prayground/shape/trianglemesh.h>
#include <filesystem>
#include <vector>

namespace prayground {

// Based on tinyobj::material_t
struct ObjMaterialParams
{
    std::string name;
};

void loadObjWithMtl(
    const std::filesystem::objpath, 
    const std::filesystem::mtlpath, 
    TriangleMesh& mesh, )

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