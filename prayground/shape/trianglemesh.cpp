#include "trianglemesh.h"
#include <prayground/core/cudabuffer.h>
#include <prayground/core/load3d.h>
#include <prayground/core/file_util.h>
#include <prayground/math/util.h>

namespace prayground {

    namespace fs = std::filesystem;

    // ------------------------------------------------------------------
    TriangleMesh::TriangleMesh()
    {

    }

    TriangleMesh::TriangleMesh(const fs::path& filename)
    {
        load(filename);
    }

    TriangleMesh::TriangleMesh(
        const std::vector<Vec3f>& vertices, 
        const std::vector<Face>& faces, 
        const std::vector<Vec3f>& normals, 
        const std::vector<Vec2f>& texcoords, 
        const std::vector<uint32_t>& sbt_indices) 
        : m_vertices(vertices), 
          m_faces(faces), 
          m_normals(normals),
          m_texcoords(texcoords), 
          m_sbt_indices(sbt_indices)
    {
    
    }

    // ------------------------------------------------------------------
    constexpr ShapeType TriangleMesh::type()
    {
        return ShapeType::Mesh;
    }

    // ------------------------------------------------------------------
    void TriangleMesh::copyToDevice() 
    {
        Data data = this->getData();

        if (!d_data) 
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(Data),
            cudaMemcpyHostToDevice
        ));
    }

    // ------------------------------------------------------------------
    OptixBuildInput TriangleMesh::createBuildInput() 
    {
        OptixBuildInput bi = {};
        CUDABuffer<uint32_t> d_sbt_indices_buf;

        d_sbt_indices_buf.copyToDevice(m_sbt_indices);
        d_sbt_indices = d_sbt_indices_buf.devicePtr();

        uint32_t num_materials = this->numMaterials();
        uint32_t* triangle_input_flags = new uint32_t[num_materials];
        for (uint32_t i = 0; i < num_materials; i++)
            triangle_input_flags[i] = OPTIX_GEOMETRY_FLAG_NONE;
    
        bi.type = static_cast<OptixBuildInputType>(this->type());
        bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        bi.triangleArray.vertexStrideInBytes = sizeof(Vec3f);
        bi.triangleArray.numVertices = static_cast<uint32_t>(m_vertices.size());
        bi.triangleArray.vertexBuffers = &d_vertices;
        bi.triangleArray.flags = triangle_input_flags;
        bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        bi.triangleArray.indexStrideInBytes = sizeof(Face);
        bi.triangleArray.numIndexTriplets = static_cast<uint32_t>(m_faces.size());
        bi.triangleArray.indexBuffer = d_faces;
        bi.triangleArray.numSbtRecords = num_materials;
        bi.triangleArray.sbtIndexOffsetBuffer = d_sbt_indices;
        bi.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        bi.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
        return bi;
    }

    // ------------------------------------------------------------------
    void TriangleMesh::free()
    {
        Shape::free();
        cuda_frees(d_vertices, d_normals, d_faces, d_texcoords);
    }

    AABB TriangleMesh::bound() const 
    {
        return AABB{};
    }

    void TriangleMesh::setSbtIndex(const uint32_t sbt_idx)
    {
        if (m_sbt_indices.size() > 1)
        {
            PG_LOG_WARN("Two or more number of indices have been already set.");
            return;
        }

        if (m_sbt_indices.size() == 1)
            m_sbt_indices.back() = sbt_idx;
        else if (m_sbt_indices.empty())
            m_sbt_indices.emplace_back(sbt_idx);
    }

    uint32_t TriangleMesh::sbtIndex() const
    {
        ASSERT(m_sbt_indices.size() != 1, "Detected invalid number of SBT indices. TriangleMesh::sbtIndex() can be called when the only ONE SBT index is set.");

        return m_sbt_indices.back();
    }

    // ------------------------------------------------------------------
    TriangleMesh::Data TriangleMesh::getData()
    {
        CUDABuffer<Vec3f> d_vertices_buf;
        CUDABuffer<Face> d_faces_buf;
        CUDABuffer<Vec3f> d_normals_buf;
        CUDABuffer<Vec2f> d_texcoords_buf;
        d_vertices_buf.copyToDevice(m_vertices);
        d_faces_buf.copyToDevice(m_faces);
        d_normals_buf.copyToDevice(m_normals);
        d_texcoords_buf.copyToDevice(m_texcoords);

        d_vertices = d_vertices_buf.devicePtr();
        d_faces = d_faces_buf.devicePtr();
        d_normals = d_normals_buf.devicePtr();
        d_texcoords = d_texcoords_buf.devicePtr();

        // device side pointer of mesh data
        Data data = {
            .vertices = d_vertices_buf.deviceData(),
            .faces = d_faces_buf.deviceData(),
            .normals = d_normals_buf.deviceData(),
            .texcoords = d_texcoords_buf.deviceData()
        };

        return data;
    }

    // ------------------------------------------------------------------
    void TriangleMesh::addVertices(const std::vector<Vec3f>& verts)
    {
        std::copy(verts.begin(), verts.end(), std::back_inserter(m_vertices));
    }

    void TriangleMesh::addFaces(const std::vector<Face>& faces)
    {
        std::copy(faces.begin(), faces.end(), std::back_inserter(m_faces));
    }

    void TriangleMesh::addFaces(const std::vector<Face>& faces, const std::vector<uint32_t>& sbt_indices)
    {
        std::copy(faces.begin(), faces.end(), std::back_inserter(m_faces));
        std::copy(sbt_indices.begin(), sbt_indices.end(), std::back_inserter(m_sbt_indices));
    }

    void TriangleMesh::addNormals(const std::vector<Vec3f>& normals)
    {
        std::copy(normals.begin(), normals.end(), std::back_inserter(m_normals));
    }

    void TriangleMesh::addTexcoords(const std::vector<Vec2f>& texcoords)
    {
        std::copy(texcoords.begin(), texcoords.end(), std::back_inserter(m_texcoords));
    }

    // ------------------------------------------------------------------
    void TriangleMesh::addVertex(const Vec3f& v)
    {
        m_vertices.emplace_back(v);
    }

    void TriangleMesh::addFace(const Face& face)
    {
        m_faces.emplace_back(face);
    }

    void TriangleMesh::addFace(const Face& face, uint32_t sbt_index)
    {
        m_faces.emplace_back(face);
        m_sbt_indices.emplace_back(sbt_index);
    }

    void TriangleMesh::addNormal(const Vec3f& n)
    {
        m_normals.emplace_back(n);
    }

    void TriangleMesh::addTexcoord(const Vec2f& texcoord)
    {
        m_texcoords.emplace_back(texcoord);
    }

    // ------------------------------------------------------------------
    void TriangleMesh::load(const fs::path& filename)
    {
        std::string ext = pgGetExtension(filename);
        if (ext == ".obj") {
            std::optional<fs::path> filepath = pgFindDataPath(filename);
            ASSERT(filepath, "The OBJ file '" + filename.string() + "' is not found.");

            pgLog("Loading OBJ file '" + filepath.value().string() + "' ...");
            loadObj(filepath.value(), m_vertices, m_faces, m_normals, m_texcoords);
        }
        else if (ext == ".ply") {
            std::optional<fs::path> filepath = pgFindDataPath(filename);
            ASSERT(filepath, "The OBJ file '" + filename.string() + "' is not found.");

            pgLog("Loading PLY file '" + filepath.value().string() + "' ...");
            loadPly(filepath.value(), m_vertices, m_faces, m_normals, m_texcoords);
        }

        // Calculate normals if they are empty.
        if (m_normals.empty())
        {
            m_normals.resize(m_faces.size());
            for (size_t i = 0; i < m_faces.size(); i++)
            {
                m_faces[i].normal_id = Vec3i(i);

                auto p0 = m_vertices[m_faces[i].vertex_id.x()];
                auto p1 = m_vertices[m_faces[i].vertex_id.y()];
                auto p2 = m_vertices[m_faces[i].vertex_id.z()];
                auto N = cross(p1 - p0, p2 - p0);
                N = length(N) != 0.0f ? normalize(N) : N;

                m_normals[i] = N;
            }
        }

        if (m_texcoords.empty()) {
            m_texcoords.resize(m_vertices.size());
        }
    }

    // ------------------------------------------------------------------
    void TriangleMesh::loadWithMtl(
        const std::filesystem::path& objpath,
        std::vector<Attributes>& material_attribs,
        const std::filesystem::path& mtlpath
    )
    {
        std::string ext = pgGetExtension(objpath);
        ASSERT(ext == ".obj", "loadObjWithMtl() only supports .obj file format with .mtl file.");

        std::optional<fs::path> filepath = pgFindDataPath(objpath);
        ASSERT(filepath, "The OBJ file '" + objpath.string() + "' is not found.");

        pgLog("Loading OBJ file '" + filepath.value().string() + "' ...");

        const size_t current_num_attribs = material_attribs.size();
        loadObjWithMtl(filepath.value(), m_vertices, m_faces, m_normals, m_texcoords, m_sbt_indices, material_attribs, mtlpath);

        // Calculate normals if they are empty.
        if (m_normals.empty())
        {
            m_normals.resize(m_faces.size());
            for (size_t i = 0; i < m_faces.size(); i++)
            {
                m_faces[i].normal_id = Vec3i(i);

                auto p0 = m_vertices[m_faces[i].vertex_id[0]];
                auto p1 = m_vertices[m_faces[i].vertex_id[1]];
                auto p2 = m_vertices[m_faces[i].vertex_id[2]];
                auto N = cross(p1 - p0, p2 - p0);
                N = length(N) != 0.0f ? normalize(N) : N;

                m_normals[i] = N;
            }
        }

        if (m_texcoords.empty()) {
            m_texcoords.resize(m_vertices.size());
        }
    }


    void TriangleMesh::smooth()
    {
        m_normals.clear();
        m_normals.resize(m_vertices.size());
        auto counts = std::vector<int>(m_vertices.size(), 0);
        for (size_t i = 0; i < m_faces.size(); i++)
        {
            m_faces[i].normal_id = m_faces[i].vertex_id;

            auto p0 = m_vertices[m_faces[i].vertex_id.x()];
            auto p1 = m_vertices[m_faces[i].vertex_id.y()];
            auto p2 = m_vertices[m_faces[i].vertex_id.z()];
            auto N = cross(p1 - p0, p2 - p0);
            N = length(N) != 0.0f ? normalize(N) : N;

            auto idx = m_faces[i].vertex_id.x();
            m_normals[idx] += N;
            counts[idx]++;

            idx = m_faces[i].vertex_id.y();
            m_normals[idx] += N;
            counts[idx]++;

            idx = m_faces[i].vertex_id.z();
            m_normals[idx] += N;
            counts[idx]++;
        }
        for (size_t i = 0; i < m_vertices.size(); i++)
        {
            m_normals[i] /= counts[i];
            m_normals[i] = normalize(m_normals[i]);
        }
    }

    void TriangleMesh::offsetSbtIndex(uint32_t sbt_base)
    {
        if (m_sbt_indices.empty())
            m_sbt_indices.push_back(sbt_base);
        else
        {
            for (auto& idx : m_sbt_indices) {
                idx += sbt_base;
            }
        }
    }

    uint32_t TriangleMesh::numMaterials() const
    {
        ASSERT(m_sbt_indices.size() != 0, "Any SBT indices aren't set");

        std::vector<uint32_t> sbt_counter;
        for (auto& sbt_idx : m_sbt_indices)
        {
            // �d�����Ȃ�index�̐��𐔂���
            auto itr = std::find(sbt_counter.begin(), sbt_counter.end(), sbt_idx);
            if (sbt_counter.empty() || itr == sbt_counter.end())
                sbt_counter.push_back(sbt_idx);
        }
        return static_cast<uint32_t>(sbt_counter.size());
    }

    void TriangleMesh::addSbtIndices(const std::vector<uint32_t>& sbt_indices)
    {
        std::copy(sbt_indices.begin(), sbt_indices.end(), std::back_inserter(m_sbt_indices));
    }

} // namespace prayground