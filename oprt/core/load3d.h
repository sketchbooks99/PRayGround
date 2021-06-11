#pragma once

#include <sutil/vec_math.h>
#include "../ext/happly/happly.h"
#include <algorithm>
#include "../shape/optix/trianglemesh.cuh"

namespace oprt {

void loadObj(
    const std::string& filename, 
    std::vector<float3>& vertices,
    std::vector<float3>& normals,
    std::vector<Face>& faces,
    std::vector<float2>& coordinates
)
{
    std::vector<float3> temp_normals;
    std::ifstream ifs(filename, std::ios::in);
    Assert(ifs.is_open(), "The OBJ file '"+filename+"' is not found.");
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
            temp_normals.emplace_back(make_float3(x, y, z));
        }
        else if (header == "f")
        {
            // temporalily vector to store face information
            std::vector<int> temp_vert_indices;
            std::vector<int> temp_norm_indices;

            // Future work -----------------------------------
            // std::vector<int> temp_tex_faces;
            // ----------------------------------------------- 
            for (std::string buffer; iss >> buffer;)
            {
                int vert_idx, tex_idx, norm_idx;
                if (sscanf(buffer.c_str(), "%d/%d/%d", &vert_idx, &tex_idx, &norm_idx) == 3)
                {
                    // Input - index(vertex)/index(texture)/index(normal)
                    temp_vert_indices.emplace_back(vert_idx - 1);
                    temp_norm_indices.emplace_back(norm_idx - 1);
                    // temp_tex_faces.emplace_back(tex_idx - 1);
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
                    //temp_tex_faces.emplace_back(tex_idx - 1);
                }
                else if (sscanf(buffer.c_str(), "%d", &vert_idx) == 1)
                {
                    // Input - index(vertex)
                    temp_vert_indices.emplace_back(vert_idx - 1);
                }
                else
                    Throw("Invalid format in face information input.");
            }
            Assert(temp_vert_indices.size() >= 3, "The number of faces is less than 3.");

            if (temp_vert_indices.size() == 3) {
                Face face{ {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
                face.vertex_id = make_int3(temp_vert_indices[0], temp_vert_indices[1], temp_vert_indices[2]);
                
                if (!temp_norm_indices.empty())
                    face.normal_id = make_int3(temp_norm_indices[0], temp_norm_indices[1], temp_norm_indices[2]);

                faces.push_back(face);
            }
            /** 
             * Get more then 4 inputs.
             * @note 
             * This case is implemented under the assumption that if face input are more than 4, 
             * mesh are configured by quad and inputs are partitioned with 4 stride. 
             */
            else if (temp_vert_indices.size() == 4)
            {
                Face face1{ {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
                Face face2{ {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };

                face1.vertex_id = make_int3(temp_vert_indices[0], temp_vert_indices[1], temp_vert_indices[2]);
                face2.vertex_id = make_int3(temp_vert_indices[2], temp_vert_indices[3], temp_vert_indices[0]);

                if (!temp_norm_indices.empty())
                {
                    face1.normal_id = make_int3(temp_norm_indices[0], temp_norm_indices[1], temp_norm_indices[2]);
                    face2.normal_id = make_int3(temp_norm_indices[2], temp_norm_indices[3], temp_norm_indices[0]);
                }
                faces.push_back(face1);
                faces.push_back(face2);
            }
        }
    }

    ifs.close();
}

void loadPly(
    const std::string& filename, 
    std::vector<float3>& vertices,
    std::vector<float3>& normals,
    std::vector<Face>& faces, 
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

    // Get face faces.
    std::vector<std::vector<size_t>> ply_faces = plyIn.getFaceIndices();
    std::transform(ply_faces.begin(), ply_faces.end(), std::back_inserter(faces), 
        [](const std::vector<size_t>& f) { 
            return Face{
                make_int3(f[0], f[1], f[2]), // vertex_id
                make_int3(0),                // normal_id
                make_int3(0)                 // texcoord_id
            }; 
        } );
}

}