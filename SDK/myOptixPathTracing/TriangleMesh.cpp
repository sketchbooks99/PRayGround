#include "TriangleMesh.h"

// At present, only ".obj" format is supported.
TriangleMesh::TriangleMesh(
    const std::string& filename, 
    float3 position, float size, float3 axis, bool isSmooth)
{
    std::vector<Normal> tmp_normals;
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
                vertices.emplace_back(x, y, z, 0.0f);
            }
            else if(header == "vn") {
                float x, y, z;
                iss >> x >> y >> z;
                x *= axis.x;
                y *= axis.y;
                z *= axis.z;
                tmp_normals.emplace_back(x, y, z, 0.0f);
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

    // Transformation
    Vertex center;
    auto min = vertices.front(), max = vertices.front();
    for (auto& vertex : vertices)
    {
        for (int i = 0; i < 3; i++)
        {
            if (vertex[i] < min[i]) min[i] = vertex[i];
            if (vertex[i] > max[i]) max[i] = vertex[i];
        }
    }

    for (auto& vertex : vertices) {
        vertex = vertex * size + position;
    }

    if(normals.empty()) {
        // Mesh smoothing
        normals.resize(vertices.size());
        auto counts = std::vector<int>(vertices.size(), 0);
        for(int i=0; i<indices.size(); i++)
        {
            auto p0 = vertices[indices[i].x];
            auto p1 = vertices[indices[i].y];
            auto p2 = vertices[indices[i].z];
            auto N = static_cast<float3>(normalize(cross(p2 - p0, p1 - p0)));

            if (isSmooth) {
                auto idx = indices[i].x;
                normals[idx] += Normal(N, 0.0f);
                counts[idx]++;
                idx = indices[i].y;
                normals[idx] += Normal(N, 0.0f);
                counts[idx]++;
                idx = indices[i].z;
                normals[idx] += Normal(N, 0.0f);
                counts[idx]++;
            }
            else
            {
                normals[indices[i].x] = Normal(N, 0.0f);
                normals[indices[i].y] = Normal(N, 0.0f);
                normals[indices[i].z] = Normal(N, 0.0f);
            }
        }
        if (isSmooth) {
            for (int i = 0; i < vertices.size(); i++)
            {
                normals[i] /= counts[i];
                normals[i] = normalize(normals[i]);
            }
        }
    } else {
        // Normals are described by obj file.
        normals.resize(vertices.size());
        for(auto i=0; i<indices.size(); i++) {
            auto idx = indices[i].x;
            normals[idx] = tmp_normals[normal_indices[i].x];
            idx = indices[i].y;
            normals[idx] = tmp_normals[normal_indices[i].y];
            idx = indices[i].z;
            normals[idx] = tmp_normals[normal_indices[i].z];
        }
    }
}

TriangleMesh::TriangleMesh(std::vector<Vertex> vertices, 
    std::vector<int3> indices, 
    std::vector<Normal> normals, 
    bool isSmooth) 
    : vertices(vertices), 
    indices(indices), 
    normals(normals)
{
    assert(vertices.size() == normals.size());

    // Mesh smoothing
    if (indices.size() > 32)
    {
        normals = std::vector<Normal>(vertices.size(), Normal());
        auto counts = std::vector<int>(vertices.size(), 0);
        for (int i = 0; i < indices.size(); i++)
        {
            auto p0 = vertices[indices[i].x];
            auto p1 = vertices[indices[i].y];
            auto p2 = vertices[indices[i].z];
            auto N = normalize(cross(p2 - p0, p1 - p0));

            auto idx = indices[i].x;
            normals[idx] += Normal(N.x, N.y, N.z, 0.0f);
            counts[idx]++;

            idx = indices[i].y;
            normals[idx] += Normal(N.x, N.y, N.z, 0.0f);
            counts[idx]++;

            idx = indices[i].z;
            normals[idx] += Normal(N.x, N.y, N.z, 0.0f);
            counts[idx]++;
        }
        for (int i = 0; i < vertices.size(); i++)
        {
            normals[i] /= counts[i];
            normals[i] = normalize(normals[i]);
        }
    }
}