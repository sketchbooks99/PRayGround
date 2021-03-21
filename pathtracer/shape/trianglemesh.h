#pragma once

#include <include/core/shape.h>
#include "optix/trianglemesh.cuh"

namespace pt {

class TriangleMesh : public Shape {
public:
    TriangleMesh() {}
    TriangleMesh(const std::string& filename, float3 position, float size, float3 axis, bool isSmooth=true);
    TriangleMesh(std::vector<float3> vertices, std::vector<int3> faces, std::vector<float3> normals, bool isSmooth=true);

    ShapeType type() const override { return ShapeType::Mesh; }

    void prepare_data() override;
    void build_input( OptixBuildInput& bi, uint32_t sbt_idx ) override;
    /**
     * @note 
     * Currently, triangle never need AABB at intersection test on the device side.
     * For future work, I'd like to make this renderer can switch a computing device 
     * (CPU/GPU) depends on an application.
     */
    AABB bound() const override { return AABB(); } 
private:
    std::vector<float3> m_vertices;
    std::vector<float3> m_normals;
    std::vector<int3> m_indices;

    CUdeviceptr d_vertices { 0 };
    CUdeviceptr d_normals { 0 };
    CUdeviceptr d_indices { 0 };
};

}
