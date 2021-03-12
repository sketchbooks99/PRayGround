#pragma once

#include <core/shape.h>

namespace pt {

#if !defined(__CUDACC__)
class TriangleMesh : public Shape {
    TriangleMesh() {}
    TriangleMesh(const std::string& filename, float3 position, float size, float3 axis, bool isSmooth=true);
    TriangleMesh(std::vector<float3> vertices, std::vector<int3> faces, std::vector<float3> normals, bool isSmooth=true);
    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<int3> indices;
};
#endif

}
