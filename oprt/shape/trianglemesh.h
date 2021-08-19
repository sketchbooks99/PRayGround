#pragma once

#include "cuda/trianglemesh.cuh"

#ifndef __CUDACC__
#include <oprt/core/shape.h>
#include <oprt/math/util.h>

namespace oprt {

/**
 * @todo Implementation of uv texcoords.
 */

class TriangleMesh final : public Shape {
public:
    TriangleMesh() {}
    explicit TriangleMesh(const std::filesystem::path& filename, bool is_smooth=true);
    TriangleMesh(
        std::vector<float3> vertices, 
        std::vector<Face> faces, 
        std::vector<float3> normals, 
        std::vector<float2> texcoords,
        bool is_smooth=true);

    OptixBuildInputType buildInputType() const override { return OPTIX_BUILD_INPUT_TYPE_TRIANGLES; }

    void copyToDevice() override;
    void buildInput( OptixBuildInput& bi ) override;
    /**
     * @note 
     * Currently, triangle never need AABB for intersection test on the device side
     * because test for triangle is built in at OptiX and automatically performed 
     * by using input mesh information.
     * However, in the future, I'd like to make this renderer be able to switch 
     * computing devices (CPU or GPU) according to the need of an application, 
     * and AABB will be needed for this.
     */
    AABB bound() const override;

    void addVertex(const float3& v);
    void addFace(const Face& face);
    void addNormal(const float3& n);
    void addTexcoord(const float2& texcoord);

    void load(const std::filesystem::path& filename, bool is_smooth=true);

    std::vector<float3> vertices() const { return m_vertices; } 
    std::vector<Face> faces() const { return m_faces; } 
    std::vector<float3> normals() const { return m_normals; }
    std::vector<float2> texcoords() const { return m_texcoords; }

    // Getters of device side pointers.
    CUdeviceptr deviceVertices() const { return d_vertices; }
    CUdeviceptr deviceFaces() const { return d_faces; }
    CUdeviceptr deviceNormals() const { return d_normals; }
    CUdeviceptr deivceTexcoords() const { return d_texcoords; }

private:
    std::vector<float3> m_vertices;
    std::vector<Face> m_faces;
    std::vector<float3> m_normals;
    std::vector<float2> m_texcoords;

    CUdeviceptr d_vertices { 0 };
    CUdeviceptr d_faces { 0 };
    CUdeviceptr d_normals { 0 };
    CUdeviceptr d_texcoords { 0 };
};

/**
 * @brief Create a quad mesh
 */
std::shared_ptr<TriangleMesh> createQuadMesh(const float u_min, const float u_max, 
                             const float v_min, const float v_max, 
                             const float k, Axis axis);

/**
 * @brief Create mesh
 */
std::shared_ptr<TriangleMesh> createTriangleMesh(const std::string& filename, bool is_smooth=true);

std::shared_ptr<TriangleMesh> createTriangleMesh(
    const std::vector<float3>& vertices,
    const std::vector<Face>& faces, 
    const std::vector<float3>& normals,
    const std::vector<float2>& texcoords,
    bool is_smooth = true
);

}

#endif
