#pragma once

#include <prayground/math/vec_math.h>
#include <filesystem>

namespace prayground {

void loadObjWithMtl(
    const std::filesystem::path& filename,
    std::vector<float3>& vertices, 
    std::vector<float3>& normals, 
    std::vector<Face>& faces, 
    std::vector<float2>& texcoords);

void loadObj(
    const std::filesystem::path& filename, 
    std::vector<float3>& vertices,
    std::vector<float3>& normals,
    std::vector<Face>& faces,
    std::vector<float2>& texcoords
);

void loadPly(
    const std::filesystem::path& filename, 
    std::vector<float3>& vertices,
    std::vector<float3>& normals,
    std::vector<Face>& faces, 
    std::vector<float2>& texcoords
);

} // ::prayground