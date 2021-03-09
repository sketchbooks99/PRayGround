#pragma once

#include <vector>
#include <sutil/vec_math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <assert.h>

#include "../core/shape.h"
#include "../core/transform.h"

namespace pt {

struct MeshData {
    float3* vertices;
    float3* normals;
    int3* indices;
    sutil::Transform transform;
};

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
