#pragma once

#include "../core/shape.h"
#include "optix/trianglemesh.cuh"

namespace pt {

/**
 * @todo Implementation of uv coordinates.
 */

class TriangleMesh : public Shape {
public:
    TriangleMesh() {}
    TriangleMesh(const std::string& filename, bool isSmooth=true);
    TriangleMesh(
        std::vector<float3> vertices, 
        std::vector<int3> faces, 
        std::vector<float3> normals, 
        std::vector<float2> coodinates,
        bool isSmooth=true);

    ShapeType type() const override { return ShapeType::Mesh; }

    void prepare_data() override;
    void build_input( OptixBuildInput& bi, uint32_t sbt_idx, unsigned int index_offset ) override;
    /**
     * \note 
     * Currently, triangle never need AABB at intersection test on the device side.
     * However, for future work, I'd like to make this renderer be able to
     * switch computing devices (CPU or GPU) according to the need of an application.
     */
    AABB bound() const override { return AABB(); } 

    std::vector<float3> vertices() const { return m_vertices; } 
    std::vector<float3> normals() const { return m_normals; }
    std::vector<int3> indices() const { return m_indices; } 
    std::vector<float2> coordinates() const { return m_coordinates; }

    /**
     * \note This is for checking if device side pointer is correctly allocated.
     */
    CUdeviceptr get_dvertices() const { return d_vertices; }
    CUdeviceptr get_dnormals() const { return d_normals; }
    CUdeviceptr get_dindices() const { return d_indices; }
    CUdeviceptr get_dcoordinates() const { return d_coordinates; }

private:
    std::vector<float3> m_vertices;
    std::vector<float3> m_normals;
    std::vector<int3> m_indices;
    std::vector<float2> m_coordinates;

    CUdeviceptr d_vertices { 0 };
    CUdeviceptr d_normals { 0 };
    CUdeviceptr d_indices { 0 };
    CUdeviceptr d_coordinates { 0 };
};

}
