#include "load3d.h"
#include <prayground/core/file_util.h>
#include <prayground/ext/happly/happly.h>

#ifndef TINEOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif
#include <prayground/ext/tinyobjloader/tiny_obj_loader.h>
#include <algorithm>

namespace prayground {

// -------------------------------------------------------------------------------
void loadObj(
    const std::filesystem::path& filepath, 
    std::vector<float3>& vertices, 
    std::vector<float3>& normals, 
    std::vector<Face>& faces, 
    std::vector<float2>& texcoords
)
{
    tinyobj::ObjReaderConfig reader_config;
    reader_config.triangulate = true; // triangulate mesh
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filepath.string(), reader_config))
    {
        ASSERT(reader.Error().empty(), "TinyObjReader: " + reader.Error());
    }

    if (!reader.Warning().empty())
        Message(MSG_WARNING, "TinyObjReader:", reader.Warning());

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();

    vertices.resize(attrib.vertices.size() / 3);
    normals.resize(attrib.normals.size() / 3);
    texcoords.resize(attrib.texcoords.size() / 2);

    for (size_t s = 0; s < shapes.size(); s++)
    {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
        {
            Face face{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            for (size_t v = 0; v < 3; v++)
            {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                setByIndex(face.vertex_id, (int)v, idx.vertex_index);
                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                vertices[idx.vertex_index] = make_float3(vx, vy, vz);

                // Normals if exists
                if (idx.normal_index >= 0)
                {
                    setByIndex(face.normal_id, (int)v, idx.normal_index);
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    normals[idx.normal_index] = make_float3(nx, ny, nz);
                }

                // Texcoords if exists
                if (idx.texcoord_index >= 0)
                {
                    setByIndex(face.texcoord_id, (int)v, idx.texcoord_index);
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    texcoords[idx.texcoord_index] = make_float2(tx, ty);
                }
            }
            faces.push_back(face);
            index_offset += 3;
        }
    }
}

// -------------------------------------------------------------------------------
void loadPly(
    const std::filesystem::path& filepath, 
    std::vector<float3>& vertices, 
    std::vector<float3>& normals, 
    std::vector<Face>& faces, 
    std::vector<float2>& texcoords
)
{
    happly::PLYData plyIn(filepath.string());
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