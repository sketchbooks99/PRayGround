#pragma once

#include <sutil/vec_math.h>
#include "../ext/happly/happly.h"
#include <algorithm>

namespace pt {

void load_obj(
    const std::string& filename, 
    std::vector<float3>& vertices,
    std::vector<float3>& normals,
    std::vector<int3>& indices,
    std::vector<float2>& coordinates
)
{
    std::vector<float3> tmp_normals;
    std::vector<int3> normal_indices;
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
            vertices.emplace_back(make_float3(x, y, z));
        }
        else if(header == "vn") {
            float x, y, z;
            iss >> x >> y >> z;
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

void load_ply(
    const std::string& filename, 
    std::vector<float3>& vertices,
    std::vector<float3>& normals,
    std::vector<int3>& indices, 
    std::vector<float2>& coordinates
) 
{
    happly::PLYData plyIn(filename);
    try {
        plyIn.validate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        Throw("The error occured while loading the PLY file.");
    }

    // Get vertices.
    std::vector<std::array<double, 3>> ply_vertices = plyIn.getVertexPositions();
    std::transform(ply_vertices.begin(), ply_vertices.end(), std::back_inserter(vertices), 
        [](const std::array<double, 3>& v) { return make_float3(v[0], v[1], v[2]); } );

    // Get face indices.
    std::vector<std::vector<size_t>> ply_faces = plyIn.getFaceIndices();
    std::transform(ply_faces.begin(), ply_faces.end(), std::back_inserter(indices), 
        [](const std::vector<size_t>& f) { return make_int3(f[0], f[1], f[2]); } );

}

}