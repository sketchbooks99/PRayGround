#include "trianglemesh.h"
#include "../core/cudabuffer.h"
#include "../core/load3d.h"

namespace pt {

/** \note At present, only .obj file format is supported. */
// ------------------------------------------------------------------
TriangleMesh::TriangleMesh(
    const std::string& filename, bool isSmooth)
{
    if (filename.substr(filename.length() - 4) == ".obj") {
        Message("Loading OBJ file '"+filename+"' ...");
        load_obj(filename, m_vertices, m_normals, m_indices, m_coordinates);
    }
    else if (filename.substr(filename.length() - 4) == ".ply") {
        Message("Loading PLY file '"+filename+"' ...");
        load_ply(filename, m_vertices, m_normals, m_indices, m_coordinates);
    }


    // Mesh smoothing
    m_normals.resize(m_vertices.size());
    auto counts = std::vector<int>(m_vertices.size(), 0);
    for(size_t i=0; i<m_indices.size(); i++)
    {
        auto p0 = m_vertices[m_indices[i].x];
        auto p1 = m_vertices[m_indices[i].y];
        auto p2 = m_vertices[m_indices[i].z];
        auto N = normalize(cross(p2 - p0, p1 - p0));

        if (isSmooth) {
            auto idx = m_indices[i].x;
            m_normals[idx] += N;
            counts[idx]++;
            idx = m_indices[i].y;
            m_normals[idx] += N;
            counts[idx]++;
            idx = m_indices[i].z;
            m_normals[idx] += N;
            counts[idx]++;
        }
        else
        {
            m_normals[m_indices[i].x] = N;
            m_normals[m_indices[i].y] = N;
            m_normals[m_indices[i].z] = N;
        }
    }
    if (isSmooth) {
        for (size_t i = 0; i < m_vertices.size(); i++)
        {
            m_normals[i] /= counts[i];
            m_normals[i] = normalize(m_normals[i]);
        }
    }
}

// ------------------------------------------------------------------
TriangleMesh::TriangleMesh(std::vector<float3> vertices, 
    std::vector<int3> indices, 
    std::vector<float3> normals, 
    bool isSmooth) 
    : m_vertices(vertices), 
      m_normals(normals), 
      m_indices(indices)
{
    Assert(m_vertices.size() == m_normals.size(), "The size of m_vertices and m_normals are not equal.");

    // Mesh smoothing
    if (m_indices.size() > 32)
    {
        m_normals = std::vector<float3>(vertices.size(), make_float3(0.0f));
        auto counts = std::vector<int>(vertices.size(), 0);
        for (size_t i = 0; i < m_indices.size(); i++)
        {
            auto p0 = m_vertices[m_indices[i].x];
            auto p1 = m_vertices[m_indices[i].y];
            auto p2 = m_vertices[m_indices[i].z];
            auto N = normalize(cross(p2 - p0, p1 - p0));

            auto idx = m_indices[i].x;
            m_normals[idx] += N;
            counts[idx]++;

            idx = m_indices[i].y;
            m_normals[idx] += N;
            counts[idx]++;

            idx = m_indices[i].z;
            m_normals[idx] += N;
            counts[idx]++;
        }
        for (size_t i = 0; i < m_vertices.size(); i++)
        {
            m_normals[i] /= counts[i];
            m_normals[i] = normalize(m_normals[i]);
        }
    }
}

// ------------------------------------------------------------------
void TriangleMesh::prepare_data() {
    CUDABuffer<float3> d_vertices_buf;
    CUDABuffer<float3> d_normals_buf;
    CUDABuffer<int3> d_indices_buf;
    d_vertices_buf.alloc_copy(m_vertices);
    d_normals_buf.alloc_copy(m_normals);
    d_indices_buf.alloc_copy(m_indices);

    // device side pointer of mesh data
    MeshData data = {
        d_vertices_buf.data(),
        d_normals_buf.data(),
        d_indices_buf.data()
    };

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(MeshData)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_data),
        &data, sizeof(MeshData),
        cudaMemcpyHostToDevice
    ));

    d_vertices = d_vertices_buf.d_ptr();
    d_normals = d_normals_buf.d_ptr();
    d_indices = d_indices_buf.d_ptr();
}   

// ------------------------------------------------------------------
void TriangleMesh::build_input( OptixBuildInput& bi, const uint32_t sbt_idx, unsigned int index_offset ) {
    CUDABuffer<uint32_t> d_sbt_indices;
    std::vector<uint32_t> sbt_indices(m_indices.size(), sbt_idx);
    d_sbt_indices.alloc_copy(sbt_indices);

    unsigned int* triangle_input_flags = new unsigned int[1];
    triangle_input_flags[0] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    
    bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    bi.triangleArray.vertexStrideInBytes = sizeof(float3);
    bi.triangleArray.numVertices = static_cast<uint32_t>(m_vertices.size());
    bi.triangleArray.vertexBuffers = &d_vertices;
    bi.triangleArray.flags = triangle_input_flags;
    bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    bi.triangleArray.indexStrideInBytes = sizeof(int3);
    bi.triangleArray.numIndexTriplets = static_cast<uint32_t>(m_indices.size());
    bi.triangleArray.indexBuffer = d_indices;
    bi.triangleArray.numSbtRecords = 1;
    bi.triangleArray.sbtIndexOffsetBuffer = d_sbt_indices.d_ptr();
    bi.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    bi.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
}

}