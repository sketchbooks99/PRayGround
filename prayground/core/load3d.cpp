#include "load3d.h"
#include <prayground/ext/happly/happly.h>
#include <prayground/shape/trianglemesh.h>
#include <algorithm>

namespace prayground {

// -------------------------------------------------------------------------------
void loadObj(
    const std::filesystem::path& filename, 
    std::vector<float3>& vertices, 
    std::vector<float3>& normals, 
    std::vector<Face>& faces, 
    std::vector<float2>& texcoords
)
{
    std::vector<float3> temp_normals;
    std::ifstream ifs(filename, std::ios::in);
    ASSERT(ifs.is_open(), "The OBJ file '" + filename.string() + "' is not found.");
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
            normals.emplace_back(make_float3(x, y, z));
        }
        else if (header == "vt")
        {
            float x, y;
            iss >> x >> y;
            texcoords.emplace_back(make_float2(x, y));
        }
        else if (header == "f")
        {
            // temporalily vector to store face information
            std::vector<int> temp_vert_indices;
            std::vector<int> temp_norm_indices;
            std::vector<int> temp_tex_indices;

            for (std::string buffer; iss >> buffer;)
            {
                int vert_idx, tex_idx, norm_idx;
                if (sscanf(buffer.c_str(), "%d/%d/%d", &vert_idx, &tex_idx, &norm_idx) == 3)
                {
                    // Input - index(vertex)/index(texture)/index(normal)
                    temp_vert_indices.emplace_back(vert_idx - 1);
                    temp_norm_indices.emplace_back(norm_idx - 1);
                    temp_tex_indices.emplace_back(tex_idx - 1);
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
                    temp_tex_indices.emplace_back(tex_idx - 1);
                }
                else if (sscanf(buffer.c_str(), "%d", &vert_idx) == 1)
                {
                    // Input - index(vertex)
                    temp_vert_indices.emplace_back(vert_idx - 1);
                }
                else
                    THROW("Invalid format in face information input.");
            }
            ASSERT(temp_vert_indices.size() >= 3, "The number of faces is less than 3.");

            if (temp_vert_indices.size() == 3) {
                Face face{ {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
                face.vertex_id = make_int3(temp_vert_indices[0], temp_vert_indices[1], temp_vert_indices[2]);
                
                if (!temp_norm_indices.empty())
                    face.normal_id = make_int3(temp_norm_indices[0], temp_norm_indices[1], temp_norm_indices[2]);
                
                if (!temp_tex_indices.empty())
                    face.texcoord_id = make_int3(temp_tex_indices[0], temp_tex_indices[1], temp_tex_indices[2]);

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

                if (!temp_tex_indices.empty())
                {
                    face1.texcoord_id = make_int3(temp_tex_indices[0], temp_tex_indices[1], temp_tex_indices[2]);
                    face2.texcoord_id = make_int3(temp_tex_indices[2], temp_tex_indices[3], temp_tex_indices[0]);
                }
                faces.push_back(face1);
                faces.push_back(face2);
            }
        }
    }

    ifs.close();
}

// -------------------------------------------------------------------------------
void loadPly(
    const std::filesystem::path& filename, 
    std::vector<float3>& vertices, 
    std::vector<float3>& normals, 
    std::vector<Face>& faces, 
    std::vector<float2>& texcoords
)
{
    happly::PLYData plyIn(filename.string());
    try {
        plyIn.validate();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        THROW("The error occured while loading the PLY file.");
    }

    // Clear arrays
    if (vertices.size()) vertices.clear();
    if (normals.size()) normals.clear();
    if (faces.size()) faces.clear();
    if (texcoords.size()) texcoords.clear();

    // Get vertices
    std::vector<std::array<double, 3>> ply_vertices = plyIn.getVertexPositions();
    std::transform(ply_vertices.begin(), ply_vertices.end(), std::back_inserter(vertices), 
        [](const std::array<double, 3>& v) { return make_float3(v[0], v[1], v[2]); } );

    // Get normals
    if (plyIn.getElement("vertex").hasProperty("nx") && 
        plyIn.getElement("vertex").hasProperty("ny") && 
        plyIn.getElement("vertex").hasProperty("nz"))
    {
        std::vector<float> x_normals = plyIn.getElement("vertex").getProperty<float>("nx");
        std::vector<float> y_normals = plyIn.getElement("vertex").getProperty<float>("ny");
        std::vector<float> z_normals = plyIn.getElement("vertex").getProperty<float>("nz");

        normals.resize(x_normals.size());
        for (size_t i = 0; auto& n : normals)
        {
            n = make_float3(x_normals[i], y_normals[i], z_normals[i]);
            i++;
        }
    }

    // Get texcoords
    if (plyIn.getElement("vertex").hasProperty("u") &&
        plyIn.getElement("vertex").hasProperty("v"))
    {
        std::vector<float> u_texcoords = plyIn.getElement("vertex").getProperty<float>("u");
        std::vector<float> v_texcoords = plyIn.getElement("vertex").getProperty<float>("v");

        texcoords.resize(u_texcoords.size());
        for (size_t i = 0; auto & texcoord : texcoords)
        {
            texcoord = make_float2(u_texcoords[i], v_texcoords[i]);
            i++;
        }
    }

    // Get faces
    std::vector<std::vector<size_t>> ply_faces = plyIn.getFaceIndices();
    std::transform(ply_faces.begin(), ply_faces.end(), std::back_inserter(faces), 
        [&](const std::vector<size_t>& f) { 
            return Face{
                make_int3(f[0], f[1], f[2]), // vertex_id
                make_int3(f[0], f[1], f[2]), // normal_id
                make_int3(f[0], f[1], f[2])  // texcoord_id
            }; 
        } );
}

} // ::prayground