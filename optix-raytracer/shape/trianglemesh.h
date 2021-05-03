#pragma once

#include "../core/shape.h"
#include "optix/trianglemesh.cuh"

namespace oprt {

/**
 * @todo Implementation of uv texcoords.
 */

class TriangleMesh final : public Shape {
public:
    TriangleMesh() {}
    explicit TriangleMesh(const std::string& filename, bool is_smooth=true);
    TriangleMesh(
        std::vector<float3> vertices, 
        std::vector<int3> indices, 
        std::vector<float3> normals, 
        std::vector<float2> texcoords,
        bool is_smooth=true);

    ShapeType type() const override { return ShapeType::Mesh; }

    void prepare_data() override;
    void build_input( OptixBuildInput& bi, uint32_t sbt_idx ) override;
    /**
     * @note 
     * Currently, triangle never need AABB at intersection test on the device side.
     * However, in the future, I'd like to make this renderer be able to
     * switch computing devices (CPU or GPU) according to the need of an application.
     */
    AABB bound() const override { return AABB(); } 

    std::vector<float3> vertices() const { return m_vertices; } 
    std::vector<int3> indices() const { return m_indices; } 
    std::vector<float3> normals() const { return m_normals; }
    std::vector<float2> texcoords() const { return m_texcoords; }

    // Getters of device side pointers.
    CUdeviceptr get_dvertices() const { return d_vertices; }
    CUdeviceptr get_dindices() const { return d_indices; }
    CUdeviceptr get_dnormals() const { return d_normals; }
    CUdeviceptr get_dtexcoords() const { return d_texcoords; }

private:
    std::vector<float3> m_vertices;
    std::vector<int3> m_indices;
    std::vector<float3> m_normals;
    std::vector<float2> m_texcoords;

    CUdeviceptr d_vertices { 0 };
    CUdeviceptr d_indices { 0 };
    CUdeviceptr d_normals { 0 };
    CUdeviceptr d_texcoords { 0 };
};

/**
 * @brief Create a quad mesh
 */
TriangleMesh* createQuadMesh(const float u_min, const float u_max, 
                             const float v_min, const float v_max, 
                             const float k, Axis axis);

/**
 * @brief Create mesh
 */
TriangleMesh* createTriangleMesh(const std::string& filename, bool is_smooth=true);

TriangleMesh* createTriangleMesh(
    const std::vector<float3>& vertices,
    const std::vector<int3>& indices, 
    const std::vector<float3>& normals,
    const std::vector<float2>& texcoords,
    bool is_smooth = true
);

}
