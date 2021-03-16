#include "trianglemesh.h"
#include <core/cudabuffer.h>

namespace pt {

/** \note At present, only .obj file format is supported. */
// ------------------------------------------------------------------
TriangleMesh::TriangleMesh(
    const std::string& filename, 
    float3 position, float size, float3 axis, bool isSmooth)
{
    std::vector<float3> tmp_normals;
    std::vector<int3> normal_indices;
    if(filename.substr(filename.length() - 4) == ".obj")
    {
        std::ifstream ifs(filename, std::ios::in);
        while (!ifs.eof())
        {
            std::string line;
            if (!std::getline(ifs, line))
                break;

            // creae string stream
            std::istringstream iss(line);
            std::string header;
            iss >> header;

            // vertex --------------------------------------
            if (header == "v")
            {
                float x, y, z;
                iss >> x >> y >> z;
                x *= axis.x;
                y *= axis.y;
                z *= axis.z;
                vertices.emplace_back(make_float3(x, y, z));
            }
            else if(header == "vn") {
                float x, y, z;
                iss >> x >> y >> z;
                x *= axis.x;
                y *= axis.y;
                z *= axis.z;
                tmp_normals.emplace_back(make_float3(x, y, z));
            }
            else if (header == "f")
            {
                // temporalily vector to store face information
                std::vector<int> temp_vert_indices;
                std::vector<int> temp_norm_indices;

                // Future work -----------------------------------
                // std::vector<int> temp_tex_indices;
                // ------------------------------------------------ 
                for (std::string buffer; iss >> buffer;)
                {
                    int vert_idx, tex_idx, norm_idx;
                    if (sscanf(buffer.c_str(), "%d/%d/%d", &vert_idx, &tex_idx, &norm_idx) == 3)
                    {
                        // Input - index(vertex)/index(texture)/index(normal)
                        temp_vert_indices.emplace_back(vert_idx - 1);
                        temp_norm_indices.emplace_back(norm_idx - 1);
                        // temp_tex_indices.emplace_back(tex_idx - 1);
                    }
                    else if (sscanf(buffer.c_str(), "%d//%d", &vert_idx, &norm_idx) == 2)
                    {
                        // Input - index(vertex)//index(normal)
                        temp_vert_indices.emplace_back(vert_idx - 1);
                        temp_norm_indices.emplace_back(norm_idx - 1);
                    }
                    else if (sscanf(buffer.c_str(), "%d/%d", &vert_idx, &tex_idx) == 2)
                    {
                        // Input - index(vertex)/index(texture)
                        temp_vert_indices.emplace_back(vert_idx - 1);
                        //temp_tex_indices.emplace_back(tex_idx - 1);
                    }
                    else if (sscanf(buffer.c_str(), "%d", &vert_idx) == 1)
                    {
                        // Input - index(vertex)
                        temp_vert_indices.emplace_back(vert_idx - 1);
                    }
                    else
                        throw std::runtime_error("Invalid format in face information input.\n");
                }
                if (temp_vert_indices.size() < 3)
                    throw std::runtime_error("The number of indices is less than 3.\n");

                if (temp_vert_indices.size() == 3) {
                    indices.emplace_back(make_int3(
                        temp_vert_indices[0], temp_vert_indices[1], temp_vert_indices[2]));
                    normal_indices.emplace_back(make_int3(
                        temp_norm_indices[0], temp_norm_indices[1], temp_norm_indices[2]));
                }
                // Get more then 4 inputs.
                // NOTE: 
                //      This case is implemented under the assumption that if face input are more than 4, 
                //      mesh are configured by quad and inputs are partitioned with 4 stride.
                else
                {
                    for (int i = 0; i<int(temp_vert_indices.size() / 4); i++)
                    {
                        // The index value of 0th vertex in quad
                        auto base_idx = i * 4;
                        indices.emplace_back(make_int3(
                            temp_vert_indices[base_idx + 0],
                            temp_vert_indices[base_idx + 1],
                            temp_vert_indices[base_idx + 2]));
                        indices.emplace_back(make_int3(
                            temp_vert_indices[base_idx + 2],
                            temp_vert_indices[base_idx + 3],
                            temp_vert_indices[base_idx + 0]));
                    }
                }
            }
        }
        ifs.close();
    }

    for (auto& vertex : vertices) {
        vertex = vertex * size + position;
    }

    // Mesh smoothing
    normals.resize(vertices.size());
    auto counts = std::vector<int>(vertices.size(), 0);
    for(int i=0; i<indices.size(); i++)
    {
        auto p0 = vertices[indices[i].x];
        auto p1 = vertices[indices[i].y];
        auto p2 = vertices[indices[i].z];
        auto N = normalize(cross(p2 - p0, p1 - p0));

        if (isSmooth) {
            auto idx = indices[i].x;
            normals[idx] += N;
            counts[idx]++;
            idx = indices[i].y;
            normals[idx] += N;
            counts[idx]++;
            idx = indices[i].z;
            normals[idx] += N;
            counts[idx]++;
        }
        else
        {
            normals[indices[i].x] = N;
            normals[indices[i].y] = N;
            normals[indices[i].z] = N;
        }
    }
    if (isSmooth) {
        for (int i = 0; i < vertices.size(); i++)
        {
            normals[i] /= counts[i];
            normals[i] = normalize(normals[i]);
        }
    }
}

// ------------------------------------------------------------------
TriangleMesh::TriangleMesh(std::vector<float3> vertices, 
    std::vector<int3> indices, 
    std::vector<float3> normals, 
    bool isSmooth) 
    : vertices(vertices), 
    indices(indices), 
    normals(normals)
{
    Assert(vertices.size() == normals.size(), "The size of vertices and normals are not equal.");

    // Mesh smoothing
    if (indices.size() > 32)
    {
        normals = std::vector<float3>(vertices.size(), make_float3(0.0f));
        auto counts = std::vector<int>(vertices.size(), 0);
        for (int i = 0; i < indices.size(); i++)
        {
            auto p0 = vertices[indices[i].x];
            auto p1 = vertices[indices[i].y];
            auto p2 = vertices[indices[i].z];
            auto N = normalize(cross(p2 - p0, p1 - p0));

            auto idx = indices[i].x;
            normals[idx] += N;
            counts[idx]++;

            idx = indices[i].y;
            normals[idx] += N;
            counts[idx]++;

            idx = indices[i].z;
            normals[idx] += N;
            counts[idx]++;
        }
        for (int i = 0; i < vertices.size(); i++)
        {
            normals[i] /= counts[i];
            normals[i] = normalize(normals[i]);
        }
    }
}

// ------------------------------------------------------------------
void TriangleMesh::prepare_data() {
    CUDABuffer<float3> d_vertices_buf;
    CUDABuffer<float3> d_normals_buf;
    CUDABuffer<int3> d_indices_buf;
    d_vertices_buf.alloc_copy(vertices);
    d_normals_buf.alloc_copy(normals);
    d_indices_buf.alloc_copy(indices);

    MeshData data = {
        d_vertices_buf.data(),
        d_normals_buf.data(),
        d_indices_buf.data(),
        Transform()
    };

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data_ptr), sizeof(MeshData)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_data_ptr),
        &data, sizeof(MeshData),
        cudaMemcpyHostToDevice
    ));
}   

// ------------------------------------------------------------------
HOST void TriangleMesh::build_input( OptixBuildInput& bi, uint32_t sbt_idx ) {
    // Prepare the indices of shader binding table on device.
    CUdeviceptr d_sbt_indices;
    std::vector<uint32_t> sbt_indices( indices.size(), sbt_idx );
    size_t size_sbt_indices_in_bytes = sbt_indices.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_indices), size_sbt_indices_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_sbt_indices),
        sbt_indices.data(),
        sbt_indices.size(),
        cudaMemcpyHostToDevice
    ));

    bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    bi.triangleArray.vertexStrideInBytes = sizeof(float3);
    bi.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    bi.triangleArray.vertexBuffers = &d_vertices;
    bi.triangleArray.flags = (unsigned int*)(OPTIX_GEOMETRY_FLAG_NONE);
    bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    bi.triangleArray.indexStrideInBytes = sizeof(int3);
    bi.triangleArray.numIndexTriplets = static_cast<uint32_t>(indices.size());
    bi.triangleArray.indexBuffer = d_indices;
    bi.triangleArray.numSbtRecords = 1;
    bi.triangleArray.sbtIndexOffsetBuffer = d_sbt_indices;
    bi.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    bi.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
}

}