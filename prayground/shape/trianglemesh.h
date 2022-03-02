#pragma once

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#include <prayground/core/util.h>
#include <prayground/core/attribute.h>
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

class TriangleMesh : public Shape {
public:
    struct Data {
        float3* vertices;
        Face* faces;
        float3* normals;
        float2* texcoords;
    };

#ifndef __CUDACC__
    TriangleMesh();
    TriangleMesh(const std::filesystem::path& filename);
    TriangleMesh(
        const std::vector<float3>& vertices, 
        const std::vector<Face>& faces, 
        const std::vector<float3>& normals, 
        const std::vector<float2>& texcoords, 
        const std::vector<uint32_t>& sbt_indices = std::vector<uint32_t>() );
    TriangleMesh(const TriangleMesh& mesh) = default;
    TriangleMesh(TriangleMesh&& mesh) = default;

    constexpr ShapeType type() override;

    OptixBuildInput createBuildInput() override;

    void copyToDevice() override;
    void free() override;

    AABB bound() const override;

    Data getData();

    /**
     * @note
     * Be careful when updating GAS/IAS after modifying the number of vertices, indices
     * because you must `rebuild` AS, not `update` 
     */
    void addVertices(const std::vector<float3>& verts);
    void addFaces(const std::vector<Face>& faces);
    void addFaces(const std::vector<Face>& faces, const std::vector<uint32_t>& sbt_indices);
    void addNormals(const std::vector<float3>& normals);
    void addTexcoords(const std::vector<float2>& texcoords);

    void addVertex(const float3& v);
    void addFace(const Face& face);
    void addFace(const Face& face, uint32_t sbt_index); // For per face materials
    void addNormal(const float3& n);
    void addTexcoord(const float2& texcoord);

    void load(const std::filesystem::path& filename);
    void loadWithMtl(
        const std::filesystem::path& objpath, 
        std::vector<Attributes>& material_attribs, 
        const std::filesystem::path& mtlpath = "");

    virtual void smooth();

    // For binding multiple materials to single mesh object
    void setPerFaceMaterial(bool is_per_face);
    void setNumMaterials(uint32_t num_materials);
    void addSbtIndices(const std::vector<uint32_t>& sbt_indices);
    void offsetSbtIndex(uint32_t sbt_base);

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

    // For binding multiple materials to single mesh object
    std::vector<uint32_t> m_sbt_indices;
    bool is_per_face_material{ false };
    uint32_t m_num_materials { 0 };

    CUdeviceptr d_vertices { 0 };
    CUdeviceptr d_faces { 0 };
    CUdeviceptr d_normals { 0 };
    CUdeviceptr d_texcoords { 0 };
    CUdeviceptr d_sbt_indices{ 0 };

#endif // __CUDACC__
};

}
