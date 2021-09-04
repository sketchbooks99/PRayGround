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

struct ObjMeshData {
    float3* vertices;
    Face* faces;
    float3* normals;
    float2* texcoords;
};

#ifndef __CUDACC__

class ObjMesh final : public Shape {
public:
    ObjMesh() {}
    explicit ObjMesh(const std::filesystem::path& filename, bool is_smooth=true);
    ObjMesh(
        std::vector<float3> vertices, 
        std::vector<int3> faces, 
        std::vector<float3> normals, 
        std::vector<float2> texcoords,
        bool is_smooth=true);

    ShapeType type() const override { return ShapeType::Mesh; }

    void copyToDevice() override;
    OptixBuildInput createBuildInput() override;

    void free() override;

    /**
     * @note 
     * GAS/IASの更新の際には注意が必要
     * 頂点数・インデックス数が変化している場合はASを更新ではなく再ビルドする必要がある
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
std::shared_ptr<ObjMesh> createQuadMesh(const float u_min, const float u_max, 
                             const float v_min, const float v_max, 
                             const float k, Axis axis);

/**
 * @brief Create mesh
 */
std::shared_ptr<ObjMesh> createObjMesh(const std::string& filename, bool is_smooth=true);

std::shared_ptr<ObjMesh> createObjMesh(
    const std::vector<float3>& vertices,
    const std::vector<Face>& faces, 
    const std::vector<float3>& normals,
    const std::vector<float2>& texcoords,
    bool is_smooth = true
);

#endif // __CUDACC__

}
