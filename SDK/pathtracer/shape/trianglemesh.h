#pragma once

#include <core/shape.h>
#include "optix/trianglemesh.cuh"

namespace pt {

class TriangleMesh : public Shape {
public:
    TriangleMesh() {}
    TriangleMesh(const std::string& filename, float3 position, float size, float3 axis, bool isSmooth=true);
    TriangleMesh(std::vector<float3> vertices, std::vector<int3> faces, std::vector<float3> normals, bool isSmooth=true);

    HOST ShapeType type() const override { return ShapeType::Mesh; }
    HOST void build_input( OptixBuildInput& bi, uint32_t sbt_idx ) const override;
private:
    void _create_ptr_on_device();

    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<int3> indices;

    CUdeviceptr d_vertices { 0 };
    CUdeviceptr d_normals { 0 };
    CUdeviceptr d_indices { 0 };
};

}
