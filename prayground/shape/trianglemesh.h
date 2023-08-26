#pragma once

#ifndef __CUDACC__
#include <prayground/core/util.h>
#include <prayground/core/attribute.h>
#include <filesystem>
#endif

#include <prayground/core/shape.h>
#include <prayground/math/vec.h>
#include <prayground/math/util.h>

#include <prayground/optix/omm.h>

namespace prayground {

    struct Face {
        Vec3i vertex_id;
        Vec3i normal_id;
        Vec3i texcoord_id;
    };

    class TriangleMesh : public Shape {
    public:
        struct Data {
            Vec3f* vertices;
            Face* faces;
            Vec3f* normals;
            Vec2f* texcoords;
        };

#ifndef __CUDACC__
        TriangleMesh();
        TriangleMesh(const std::filesystem::path& filename);
        TriangleMesh(
            const std::vector<Vec3f>& vertices, 
            const std::vector<Face>& faces, 
            const std::vector<Vec3f>& normals, 
            const std::vector<Vec2f>& texcoords, 
            const std::vector<uint32_t>& sbt_indices = std::vector<uint32_t>() );
        TriangleMesh(const TriangleMesh& mesh) = default;
        TriangleMesh(TriangleMesh&& mesh) = default;

        constexpr ShapeType type() override;

        OptixBuildInput createBuildInput() override;

        void copyToDevice() override;
        void free() override;

        uint32_t numPrimitives() const override;

        AABB bound() const override;

        void setSbtIndex(const uint32_t sbt_idx) override;
        uint32_t sbtIndex() const override;

        Data getData();

        /**
         * @note
         * Be careful when updating GAS/IAS after modifying the number of vertices, indices
         * because you must `rebuild` AS, not `update` 
         */
        void addVertices(const std::vector<Vec3f>& verts);
        void addFaces(const std::vector<Face>& faces);
        void addFaces(const std::vector<Face>& faces, const std::vector<uint32_t>& sbt_indices);
        void addNormals(const std::vector<Vec3f>& normals);
        void addTexcoords(const std::vector<Vec2f>& texcoords);

        void addVertex(const Vec3f& v);
        void addVertex(float x, float y, float z);
        void addFace(const Face& face);
        void addFace(const Face& face, uint32_t sbt_index); // For per face materials
        void addNormal(const Vec3f& n);
        void addNormal(float x, float y, float z);
        void addTexcoord(const Vec2f& texcoord);
        void addTexcoord(float x, float y);

        void load(const std::filesystem::path& filename);
        void loadWithMtl(
            const std::filesystem::path& objpath, 
            std::vector<Attributes>& material_attribs, 
            const std::filesystem::path& mtlpath = ""
        );

        /* Calculate normals based on triangle faces. The normals are stored for each faces, 
         * so more memory size will be required than smoothed normals. */
        void calculateNormalFlat();
        /* Calculate smooth normals for all vertices. The number of vertices and normals is same. */
        void calculateNormalSmooth();

        /* For binding multiple materials to single mesh object */
        void setSbtIndices(const std::vector<uint32_t>& sbt_indices);
        void offsetSbtIndex(uint32_t sbt_base);
        uint32_t numMaterials() const;

        const Vec3f& vertexAt(const int32_t i) const;
        const Vec3f& normalAt(const int32_t i) const;
        const Face& faceAt(const int32_t i) const;
        const Vec2f& texcoordAt(const int32_t i) const;

        const std::vector<Vec3f>& vertices() const { return m_vertices; }
        const std::vector<Face>& faces() const { return m_faces; }
        const std::vector<Vec3f>& normals() const { return m_normals; }
        const std::vector<Vec2f>& texcoords() const { return m_texcoords; }
        const std::vector<uint32_t>& sbtIndices() const { return m_sbt_indices; }

        const uint32_t& numVertices() const { return static_cast<uint32_t>(m_vertices.size()); }
        const uint32_t& numFaces() const { return static_cast<uint32_t>(m_faces.size()); }
        const uint32_t& numNormals() const { return static_cast<uint32_t>(m_normals.size()); }
        const uint32_t& numTexcoords() const { return static_cast<uint32_t>(m_texcoords.size()); }
        const uint32_t& numSbtIndices() const { return static_cast<uint32_t>(m_sbt_indices.size()); }

        CUdeviceptr deviceVertices() const { return d_vertices; }
        CUdeviceptr deviceFaces() const { return d_faces; }
        CUdeviceptr deviceNormals() const { return d_normals; }
        CUdeviceptr deivceTexcoords() const { return d_texcoords; }
        CUdeviceptr deviceSbtIndices() const { return d_sbt_indices; }

    protected:
        std::vector<Vec3f> m_vertices;
        std::vector<Face> m_faces;
        std::vector<Vec3f> m_normals;
        std::vector<Vec2f> m_texcoords;
        std::vector<uint32_t> m_sbt_indices;

        CUdeviceptr d_vertices { 0 };
        CUdeviceptr d_faces { 0 };
        CUdeviceptr d_normals { 0 };
        CUdeviceptr d_texcoords { 0 };
        CUdeviceptr d_sbt_indices{ 0 };

#if OPTIX_VERSION >= 70600
        OpacityMicromap m_opacitymap;
#elif OPTIX_VERSION >= 70700
#endif

#endif // __CUDACC__
    };

#ifndef __CUDACC__
    inline std::ostream& operator<<(std::ostream& out, const Face& face)
    {
        out << "Face: {" << std::endl;
        out << "\tvertex_id: " << face.vertex_id << "," << std::endl;
        out << "\tnormal_id: " << face.normal_id << "," << std::endl;
        out << "\ttexcoord_id: " << face.texcoord_id << "," << std::endl;
        out << "}";
        return out;
    }
#endif

} // namespace prayground
