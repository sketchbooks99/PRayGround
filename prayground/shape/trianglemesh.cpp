#include "trianglemesh.h"
#include <prayground/core/cudabuffer.h>
#include <prayground/core/load3d.h>
#include <prayground/core/file_util.h>
#include <prayground/math/util.h>

namespace prayground {

namespace fs = std::filesystem;

/** @note At present, only .obj file format is supported. */
// ------------------------------------------------------------------
TriangleMesh::TriangleMesh(const fs::path& filename)
{
    load(filename);
}

TriangleMesh::TriangleMesh(
    std::vector<float3> vertices, 
    std::vector<Face> faces, 
    std::vector<float3> normals, 
    std::vector<float2> texcoords) 
    : m_vertices(vertices), 
      m_faces(faces), 
      m_normals(normals),
      m_texcoords(texcoords)
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
    MeshData data = this->deviceData();

    if (!d_data) 
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(MeshData)));
    CUDA_CHECK(cudaMemcpy(
        d_data,
        &data, sizeof(MeshData),
        cudaMemcpyHostToDevice
    ));
}

// ------------------------------------------------------------------
OptixBuildInput TriangleMesh::createBuildInput() 
{
    OptixBuildInput bi = {};
    CUDABuffer<uint32_t> d_sbt_indices;
    uint32_t* sbt_indices = new uint32_t[1];
    sbt_indices[0] = m_sbt_index;
    d_sbt_indices.copyToDevice(sbt_indices, sizeof(uint32_t));

    unsigned int* triangle_input_flags = new unsigned int[1];
    triangle_input_flags[0] = OPTIX_GEOMETRY_FLAG_NONE;
    
    bi.type = static_cast<OptixBuildInputType>(this->type());
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    bi.triangleArray.vertexStrideInBytes = sizeof(float3);
    bi.triangleArray.numVertices = static_cast<uint32_t>(m_vertices.size());
    bi.triangleArray.vertexBuffers = &d_vertices;
    bi.triangleArray.flags = triangle_input_flags;
    bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    bi.triangleArray.indexStrideInBytes = sizeof(Face);
    bi.triangleArray.numIndexTriplets = static_cast<uint32_t>(m_faces.size());
    bi.triangleArray.indexBuffer = d_faces;
    bi.triangleArray.numSbtRecords = 1;
    bi.triangleArray.sbtIndexOffsetBuffer = d_sbt_indices.devicePtr();
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

// ------------------------------------------------------------------
TriangleMesh::DataType TriangleMesh::deviceData()
{
    CUDABuffer<float3> d_vertices_buf;
    CUDABuffer<Face> d_faces_buf;
    CUDABuffer<float3> d_normals_buf;
    CUDABuffer<float2> d_texcoords_buf;
    d_vertices_buf.copyToDevice(m_vertices);
    d_faces_buf.copyToDevice(m_faces);
    d_normals_buf.copyToDevice(m_normals);
    d_texcoords_buf.copyToDevice(m_texcoords);

    d_vertices = d_vertices_buf.devicePtr();
    d_faces = d_faces_buf.devicePtr();
    d_normals = d_normals_buf.devicePtr();
    d_texcoords = d_texcoords_buf.devicePtr();

    // device side pointer of mesh data
    MeshData data = {
        .vertices = d_vertices_buf.deviceData(),
        .faces = d_faces_buf.deviceData(),
        .normals = d_normals_buf.deviceData(),
        .texcoords = d_texcoords_buf.deviceData()
    };

    return data;
}

// ------------------------------------------------------------------
void TriangleMesh::addVertex(const float3& v)
{
    TODO_MESSAGE();
}

void TriangleMesh::addFace(const Face& face)
{
    TODO_MESSAGE();
}

void TriangleMesh::addNormal(const float3& n)
{
    TODO_MESSAGE();
}

void TriangleMesh::addTexcoord(const float2& texcoord)
{
    TODO_MESSAGE();
}

// ------------------------------------------------------------------
void TriangleMesh::load(const fs::path& filename)
{
    if (filename.string().substr(filename.string().length() - 4) == ".obj") {
        std::optional<fs::path> filepath = findDataPath(filename);
        ASSERT(filepath, "The OBJ file '" + filename.string() + "' is not found.");

        Message(MSG_NORMAL, "Loading OBJ file '" + filepath.value().string() + "' ...");
        loadObj(filepath.value(), m_vertices, m_normals, m_faces, m_texcoords);
    }
    else if (filename.string().substr(filename.string().length() - 4) == ".ply") {
        std::optional<fs::path> filepath = findDataPath(filename);
        ASSERT(filepath, "The OBJ file '" + filename.string() + "' is not found.");
            
        Message(MSG_NORMAL, "Loading PLY file '" + filepath.value().string() + "' ...");
        loadPly(filepath.value(), m_vertices, m_normals, m_faces, m_texcoords);
    }

    // Calculate normals if they are empty.
    if (m_normals.empty())
    {
        m_normals.resize(m_faces.size());
        for(size_t i=0; i<m_faces.size(); i++)
        {
            m_faces[i].normal_id = make_int3(i);

            auto p0 = m_vertices[m_faces[i].vertex_id.x];
            auto p1 = m_vertices[m_faces[i].vertex_id.y];
            auto p2 = m_vertices[m_faces[i].vertex_id.z];
            auto N = normalize(cross(p1 - p0, p2 - p0));

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
    m_normals = std::vector<float3>(m_vertices.size(), make_float3(0.0f));
    auto counts = std::vector<int>(m_vertices.size(), 0);
    for (size_t i = 0; i < m_faces.size(); i++)
    {
        m_faces[i].normal_id = m_faces[i].vertex_id;

        auto p0 = m_vertices[m_faces[i].vertex_id.x];
        auto p1 = m_vertices[m_faces[i].vertex_id.y];
        auto p2 = m_vertices[m_faces[i].vertex_id.z];
        auto N = normalize(cross(p0 - p1, p0 - p2));

        auto idx = m_faces[i].vertex_id.x;
        m_normals[idx] += N;
        counts[idx]++;

        idx = m_faces[i].vertex_id.y;
        m_normals[idx] += N;
        counts[idx]++;

        idx = m_faces[i].vertex_id.z;
        m_normals[idx] += N;
        counts[idx]++;
    }
    for (size_t i = 0; i < m_vertices.size(); i++)
    {
        m_normals[i] /= counts[i];
        m_normals[i] = normalize(m_normals[i]);
    }
}

} // ::prayground