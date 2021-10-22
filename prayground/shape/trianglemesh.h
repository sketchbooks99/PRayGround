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

class TriangleMesh : public Shape {
public:
    using DataType = MeshData;

    TriangleMesh();
    TriangleMesh(const std::filesystem::path& filename);
    TriangleMesh(
        const std::vector<float3>& vertices, 
        const std::vector<Face>& faces, 
        const std::vector<float3>& normals, 
        const std::vector<float2>& texcoords);
    TriangleMesh(const TriangleMesh& mesh) = default;
    TriangleMesh(TriangleMesh&& mesh) = default;

    constexpr ShapeType type() override;

    OptixBuildInput createBuildInput() override;

    void copyToDevice() override;
    void free() override;

    AABB bound() const override;

    DataType deviceData();

    /**
     * @note
     * Be careful when updating GAS/IAS after modifying the number of vertices, indices
     * because you must `rebuild` AS, not `update` 
     */
    void addVertex(const float3& v);
    void addFace(const Face& face);
    void addNormal(const float3& n);
    void addTexcoord(const float2& texcoord);

    void load(const std::filesystem::path& filename);

    virtual void smooth();

    std::vector<float3> vertices() const { return m_vertices; } 
    std::vector<Face> faces() const { return m_faces; } 
    std::vector<float3> normals() const { return m_normals; }
    std::vector<float2> texcoords() const { return m_texcoords; }

    CUdeviceptr deviceVertices() const { return d_vertices; }
    CUdeviceptr deviceFaces() const { return d_faces; }
    CUdeviceptr deviceNormals() const { return d_normals; }
    CUdeviceptr deivceTexcoords() const { return d_texcoords; }

protected:
    std::vector<float3> m_vertices;
    std::vector<Face> m_faces;
    std::vector<float3> m_normals;
    std::vector<float2> m_texcoords;

    CUdeviceptr d_vertices { 0 };
    CUdeviceptr d_faces { 0 };
    CUdeviceptr d_normals { 0 };
    CUdeviceptr d_texcoords { 0 };
};

#endif // __CUDACC__

}
