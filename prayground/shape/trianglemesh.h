#pragma once

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#include <prayground/core/util.h>
#include <filesystem>
#endif

#include <prayground/math/vec_math.h>
#include <prayground/math/util.h>

namespace prayground {

struct Face {
    int3 vertex_id; 
    int3 normal_id; 
    int3 texcoord_id;
};

struct MeshData {
    float3* vertices;
    Face* faces;
    float3* normals;
    float2* texcoords;
};

#ifndef __CUDACC__

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

    ShapeType type() const override;

    void copyToDevice() override;
    OptixBuildInput createBuildInput() override;

    void free() override;

    /**
     * @note
     * Be careful when updating GAS/IAS after modifying the number of vertices, indices
     * because you must `rebuild` AS, not `update` 
     */
    void addVertex(const float3& v);
    void addFace(const Face& face);
    void addNormal(const float3& n);
    void addTexcoord(const float2& texcoord);

    void load(const std::filesystem::path& filename, bool is_smooth=true);

    std::vector<float3> vertices() const { return m_vertices; } 
    std::vector<Face> faces() const { return m_faces; } 
    std::vector<float3> normals() const { return m_normals; }
    std::vector<float2> texcoords() const { return m_texcoords; }

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

#endif // __CUDACC__

}
