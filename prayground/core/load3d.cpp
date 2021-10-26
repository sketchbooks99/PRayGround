#include "load3d.h"
#include <prayground/core/file_util.h>
#include <prayground/ext/happly/happly.h>
#include <algorithm>

#ifndef TINEOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif
#include <prayground/ext/tinyobjloader/tiny_obj_loader.h>

namespace prayground {

// -------------------------------------------------------------------------------
void loadObj(
    const std::filesystem::path& filepath, 
    std::vector<float3>& vertices,
    std::vector<Face>& faces, 
    std::vector<float3>& normals,  
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

void loadObj(
    const std::filesystem::path& filepath, 
    TriangleMesh& mesh
)
{
    mesh.load(filepath);
}

// -------------------------------------------------------------------------------
void loadObjWithMtl(
    const std::filesystem::path& objpath, 
    std::vector<float3>& vertices,
    std::vector<Face>& faces, 
    std::vector<float3>& normals,  
    std::vector<float2>& texcoords, 
    std::vector<uint32_t>& face_indices,
    std::vector<Attributes>& material_attribs, 
    const std::filesystem::path& mtlpath = ""
)
{
    tinyobj::ObjReaderConfig reader_config;
    // trianglulate mesh
    reader_config.triangulate = true; 
    // .mth filepath
    std::string mtl_dir = getDir(objpath).string();
    if (mtlpath.string() != "")
        reader_config.mtl_search_path = getDir(mtlpath).string();

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(objpath.string(), reader_config))
    {
        ASSERT(reader.Error().empty(), "TinyObjReader: " + reader.Error());
    }

    if (!reader.Warning().empty())
        Message(MSG_WARNING, "TinyObjReader:", reader.Warning());

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

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

            face_indices.emplace_back(shapes[s].mesh.material_ids[f]);
        }
    }
    
    for (const auto& m : materials)
    {
        Attributes attrib;
        attrib.name = m.name;
        float3* ambient = new float3;
        float3* diffuse = new float3;
        float3* specular = new float3;
        float3* transmittance = new float3;
        float3* emission = new float3;
        *ambient = make_float3(m.ambient[0], m.ambient[1], m.ambient[2]);
        *diffuse = make_float3(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
        *specular = make_float3(m.specular[0], m.specular[1], m.specular[2]);
        *transmittance = make_float3(m.transmittance[0], m.transmittance[1], m.transmittance[2]);
        *emission = make_float3(m.emission[0], m.emission[1], m.emission[2]);

        attrib.addFloat3("ambient", std::unique_ptr<float3[]>(ambient), 1);
        attrib.addFloat3("diffuse", std::unique_ptr<float3[]>(diffuse), 1);
        attrib.addFloat3("specular", std::unique_ptr<float3[]>(specular), 1);
        attrib.addFloat3("transmittance", std::unique_ptr<float3[]>(transmittance), 1);
        attrib.addFloat3("emission", std::unique_ptr<float3[]>(emission), 1);

        float* shininess = new float(m.shininess);
        float* ior = new float(m.ior);
        float* dissolve = new float(m.dissolve);
        attrib.addFloat("shininess", std::unique_ptr<float[]>(shininess), 1);
        attrib.addFloat("ior", std::unique_ptr<float[]>(ior), 1);
        attrib.addFloat("dissolve", std::unique_ptr<float[]>(dissolve), 1);

        if (!m.ambient_texname.empty())
            attrib.addString("diffuse_texture", std::unique_ptr<std::string[]>(new std::string(pathJoin(mtl_dir, m.diffuse_texname))), 1);
        if (!m.diffuse_texname.empty())
            attrib.addString("diffuse_texture", std::unique_ptr<std::string[]>(new std::string(pathJoin(mtl_dir, m.diffuse_texname))), 1);
        if (!m.specular_texname.empty())
            attrib.addString("specular_texture", std::unique_ptr<std::string[]>(new std::string(pathJoin(mtl_dir, m.specular_texname))), 1);
        if (!m.specular_highlight_texname.empty())
            attrib.addString("specular_highlight_texture", std::unique_ptr<std::string[]>(new std::string(pathJoin(mtl_dir, m.specular_highlight_texname))), 1);
        if (!m.bump_texname.empty())
            attrib.addString("bump_texture", std::unique_ptr<std::string[]>(new std::string(pathJoin(mtl_dir, m.bump_texname))), 1);
        if (!m.displacement_texname.empty())
            attrib.addString("displacement_texture", std::unique_ptr<std::string[]>(new std::string(pathJoin(mtl_dir, m.displacement_texname))), 1);
        if (!m.alpha_texname.empty())
            attrib.addString("alpha_texture", std::unique_ptr<std::string[]>(new std::string(pathJoin(mtl_dir, m.alpha_texname))), 1);
        if (!m.reflection_texname.empty())
            attrib.addString("reflection_texture", std::unique_ptr<std::string[]>(new std::string(pathJoin(mtl_dir, m.reflection_texname))), 1);

        material_attribs.emplace_back(attrib);
    }
}

void loadObjWithMtl(
    const std::filesystem::path& objpath, 
    const std::filesystem::path& mtlpath, 
    TriangleMesh& mesh, 
    std::vector<Attributes>& material_attribs
)
{
    std::vector<float3> vertices;
    std::vector<Face> faces;
    std::vector<float3> normals;
    std::vector<float2> texcoords;
    std::vector<uint32_t> face_indices;

    loadObjWithMtl(objpath, vertices, faces, normals, texcoords, face_indices, material_attribs, mtlpath);
    mesh.addVertices(vertices);
    mesh.addFaces(faces, face_indices);
    mesh.addNormals(normals);
    mesh.addTexcoords(texcoords);
}

void loadObjWithMtl(
    const std::filesystem::path& filepath, 
    TriangleMesh& mesh, 
    std::vector<Attributes>& material_attribs
)
{
    std::vector<float3> vertices;
    std::vector<Face> faces;
    std::vector<float3> normals;
    std::vector<float2> texcoords;
    std::vector<uint32_t> face_indices;

    loadObjWithMtl(filepath, vertices, faces, normals, texcoords, face_indices, material_attribs);
    mesh.addVertices(vertices);
    mesh.addFaces(faces, face_indices);
    mesh.addNormals(normals);
    mesh.addTexcoords(texcoords);
}

// -------------------------------------------------------------------------------
void loadPly(
    const std::filesystem::path& filepath, 
    std::vector<float3>& vertices,
    std::vector<Face>& faces,
    std::vector<float3>& normals, 
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