#pragma once

#include <prayground/math/vec_math.h>
#include <prayground/shape/trianglemesh.h>
#include <filesystem>
#include <vector>

namespace prayground {

void loadObjWithMtl(
    const std::filesystem::path& filepath,
    std::vector<float3>& vertices, 
    std::vector<float3>& normals, 
    std::vector<Face>& faces, 
    std::vector<float2>& texcoords);

void loadObjWithMtl(
    const std::filesystem::path& objpath, 
    const std::filesystem::path& mtlpath, 
    std::vector<float3>& vertices, 
    std::vector<float3>& normals, 
    std::vector<Face>& faces, 
    std::vector<float2>& texcoords
);

void loadObj(
    const std::filesystem::path& filepath,
    TriangleMesh& mesh
);

void loadObj(
    const std::filesystem::path& filepath, 
    std::vector<float3>& vertices,
    std::vector<Face>& faces,
    std::vector<float3>& normals,
    std::vector<float2>& texcoords
);

void loadPly(
    const std::filesystem::path& filepath, 
    std::vector<float3>& vertices,
    std::vector<Face>& faces, 
    std::vector<float3>& normals,
    std::vector<float2>& texcoords
);

} // ::prayground