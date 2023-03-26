#include "load3d.h"
#include <prayground/core/file_util.h>
#include <algorithm>

// HapPLY
#include <happly/happly.h>

// tinyobjloader
#ifndef TINEOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif
#include <tinyobjloader/tiny_obj_loader.h>

// NanoVDB
#include <nanovdb/util/IO.h>

// TinyUSDZ
#include <tinyusdz/src/tinyusdz.hh>
#include <tinyusdz/src/tydra/render-data.hh>
#include <tinyusdz/src/tydra/scene-access.hh>
#include <tinyusdz/src/tydra/shader-network.hh>
#include <tinyusdz/src/usdShade.hh>
#include <tinyusdz/src/pprinter.hh>
#include <tinyusdz/src/prim-pprint.hh>
#include <tinyusdz/src/value-pprint.hh>
#include <tinyusdz/src/value-types.hh>

namespace prayground {

    namespace fs = std::filesystem;

    // -------------------------------------------------------------------------------
    void loadObj(
        const fs::path& filepath, 
        std::vector<Vec3f>& vertices,
        std::vector<Face>& faces, 
        std::vector<Vec3f>& normals,  
        std::vector<Vec2f>& texcoords
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
            pgLogWarn("TinyObjReader:", reader.Warning());

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
                    face.vertex_id[v] = idx.vertex_index;
                    tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                    tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                    tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                    vertices[idx.vertex_index] = Vec3f(vx, vy, vz);

                    // Normals if exists
                    if (idx.normal_index >= 0)
                    {
                        face.normal_id[v] = idx.normal_index;
                        tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                        tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                        tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                        normals[idx.normal_index] = Vec3f(nx, ny, nz);
                    }

                    // Texcoords if exists
                    if (idx.texcoord_index >= 0)
                    {
                        face.texcoord_id[v] = idx.texcoord_index;
                        tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                        tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                        texcoords[idx.texcoord_index] = Vec2f(tx, ty);
                    }
                }
                faces.push_back(face);
                index_offset += 3;
            }
        }
    }

    void loadObj(
        const fs::path& filepath, 
        TriangleMesh& mesh
    )
    {
        mesh.load(filepath);
    }

    // -------------------------------------------------------------------------------
    void loadObjWithMtl(
        const fs::path& objpath, 
        std::vector<Vec3f>& vertices,
        std::vector<Face>& faces, 
        std::vector<Vec3f>& normals,  
        std::vector<Vec2f>& texcoords, 
        std::vector<uint32_t>& face_indices,
        std::vector<Attributes>& material_attribs, 
        const fs::path& mtlpath = ""
    )
    {
        tinyobj::ObjReaderConfig reader_config;
        // trianglulate mesh
        reader_config.triangulate = true; 
        // .mth filepath
        std::string mtl_dir = pgGetDir(objpath).string();
        if (mtlpath.string() != "")
            reader_config.mtl_search_path = pgGetDir(mtlpath).string();

        tinyobj::ObjReader reader;
        if (!reader.ParseFromFile(objpath.string(), reader_config))
        {
            ASSERT(reader.Error().empty(), "TinyObjReader: " + reader.Error());
        }

        if (!reader.Warning().empty())
            pgLogWarn("TinyObjReader:", reader.Warning());

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
                    face.vertex_id[v] = idx.vertex_index;
                    tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                    tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                    tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                    vertices[idx.vertex_index] = Vec3f(vx, vy, vz);

                    // Normals if exists
                    if (idx.normal_index >= 0)
                    {
                        face.normal_id[v] = idx.normal_index;
                        tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                        tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                        tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                        normals[idx.normal_index] = Vec3f(nx, ny, nz);
                    }

                    // Texcoords if exists
                    if (idx.texcoord_index >= 0)
                    {
                        face.texcoord_id[v] = idx.texcoord_index;
                        tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                        tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                        texcoords[idx.texcoord_index] = Vec2f(tx, ty);
                    }
                }
                faces.push_back(face);
                index_offset += 3;

                face_indices.emplace_back(shapes[s].mesh.material_ids[f]);
            }
        }

        auto addTexture = [&](Attributes& attrib, const std::string& name, const std::string& tex_name) -> void
        {
            if (!tex_name.empty()) {
                std::unique_ptr<std::string[]> str(new std::string[1]);
                str[0] = pgPathJoin(mtl_dir, tex_name).string();
                attrib.addString(name, std::move(str), 1);
            }
        };
    
        for (const auto& m : materials)
        {
            Attributes attrib;
            attrib.name = m.name;
            Vec3f* ambient = new Vec3f;
            Vec3f* diffuse = new Vec3f;
            Vec3f* specular = new Vec3f;
            Vec3f* transmittance = new Vec3f;
            Vec3f* emission = new Vec3f;
            *ambient = Vec3f(m.ambient[0], m.ambient[1], m.ambient[2]);
            *diffuse = Vec3f(m.diffuse[0], m.diffuse[1], m.diffuse[2]);
            *specular = Vec3f(m.specular[0], m.specular[1], m.specular[2]);
            *transmittance = Vec3f(m.transmittance[0], m.transmittance[1], m.transmittance[2]);
            *emission = Vec3f(m.emission[0], m.emission[1], m.emission[2]);

            attrib.addVec3f("ambient", std::unique_ptr<Vec3f[]>(ambient), 1);
            attrib.addVec3f("diffuse", std::unique_ptr<Vec3f[]>(diffuse), 1);
            attrib.addVec3f("specular", std::unique_ptr<Vec3f[]>(specular), 1);
            attrib.addVec3f("transmittance", std::unique_ptr<Vec3f[]>(transmittance), 1);
            attrib.addVec3f("emission", std::unique_ptr<Vec3f[]>(emission), 1);

            float* shininess = new float(m.shininess);
            float* ior = new float(m.ior);
            float* dissolve = new float(m.dissolve);
            attrib.addFloat("shininess", std::unique_ptr<float[]>(shininess), 1);
            attrib.addFloat("ior", std::unique_ptr<float[]>(ior), 1);
            attrib.addFloat("dissolve", std::unique_ptr<float[]>(dissolve), 1);

            addTexture(attrib, "ambient_texture", m.ambient_texname);
            addTexture(attrib, "diffuse_texture", m.diffuse_texname);
            addTexture(attrib, "specular_texture", m.specular_texname);
            addTexture(attrib, "specular_highlight_texture", m.specular_highlight_texname);
            addTexture(attrib, "bump_texture", m.bump_texname);
            addTexture(attrib, "displacement_texture", m.displacement_texname);
            addTexture(attrib, "alpha_texture", m.alpha_texname);
            addTexture(attrib, "reflection_texture", m.reflection_texname);

            material_attribs.emplace_back(attrib);
        }
    }

    void loadObjWithMtl(
        const fs::path& objpath, 
        const fs::path& mtlpath, 
        TriangleMesh& mesh, 
        std::vector<Attributes>& material_attribs
    )
    {
        std::vector<Vec3f> vertices;
        std::vector<Face> faces;
        std::vector<Vec3f> normals;
        std::vector<Vec2f> texcoords;
        std::vector<uint32_t> face_indices;

        loadObjWithMtl(objpath, vertices, faces, normals, texcoords, face_indices, material_attribs, mtlpath);
        mesh.addVertices(vertices);
        mesh.addFaces(faces, face_indices);
        mesh.addNormals(normals);
        mesh.addTexcoords(texcoords);
    }

    void loadObjWithMtl(
        const fs::path& filepath, 
        TriangleMesh& mesh, 
        std::vector<Attributes>& material_attribs
    )
    {
        std::vector<Vec3f> vertices;
        std::vector<Face> faces;
        std::vector<Vec3f> normals;
        std::vector<Vec2f> texcoords;
        std::vector<uint32_t> face_indices;

        loadObjWithMtl(filepath, vertices, faces, normals, texcoords, face_indices, material_attribs);
        mesh.addVertices(vertices);
        mesh.addFaces(faces, face_indices);
        mesh.addNormals(normals);
        mesh.addTexcoords(texcoords);
    }

    // -------------------------------------------------------------------------------
    void loadPly(
        const fs::path& filepath, 
        std::vector<Vec3f>& vertices,
        std::vector<Face>& faces,
        std::vector<Vec3f>& normals, 
        std::vector<Vec2f>& texcoords
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
            [](const std::array<double, 3>& v) { return Vec3f(v[0], v[1], v[2]); } );

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
                n = Vec3f(x_normals[i], y_normals[i], z_normals[i]);
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
                texcoord = Vec2f(u_texcoords[i], v_texcoords[i]);
                i++;
            }
        }

        // Get faces
        std::vector<std::vector<size_t>> ply_faces = plyIn.getFaceIndices();
        std::transform(ply_faces.begin(), ply_faces.end(), std::back_inserter(faces), 
            [&](const std::vector<size_t>& f) { 
                return Face{
                    Vec3i(f[0], f[1], f[2]), // vertex_id
                    Vec3i(f[0], f[1], f[2]), // normal_id
                    Vec3i(f[0], f[1], f[2])  // texcoord_id
                }; 
            } );
    }

    // -------------------------------------------------------------------------------
    void loadNanoVDB(const fs::path& filepath, nanovdb::GridHandle<>& handle)
    {
        nanovdb::GridHandle<> nano_handle;

        auto list = nanovdb::io::readGridMetaData(filepath.string());
        for (auto& m : list)
            pgLog("       ", m.gridName);
        ASSERT(list.size() > 0, "The grid data is not found or incorrect.");

        /* Create grid */
        std::string first_gridname = list[0].gridName;

        if (first_gridname.length() > 0)
            nano_handle = nanovdb::io::readGrid<>(filepath.string(), first_gridname);
        else
            nano_handle = nanovdb::io::readGrid<>(filepath.string());

        if (!nano_handle)
        {
            std::stringstream ss;
            ss << "Unable to read " << first_gridname << " from " << filepath.string();
            THROW(ss.str());
        }

        auto* meta_data = nano_handle.gridMetaData();
        if (meta_data->isPointData())
            THROW("NanoVDB Point Data cannot be handled by PRayGround.");
        if (meta_data->isLevelSet())
            THROW("NanoVDB Level Sets cannot be handled by PRayGround.");

        ASSERT(nano_handle.size() != 0, "The size of grid data is zero.");

        handle = std::move(nano_handle);
    }

    // -------------------------------------------------------------------------------
    void loadUSD(
        const std::filesystem::path& filepath, 
        std::vector<Vec3f>& vertices, 
        std::vector<Face>& faces, 
        std::vector<Vec3f>& normals, 
        std::vector<Vec2f>& texcoords)
    {
        std::string warn;
        std::string err;

        std::string ext = pgGetLowerString(pgGetExtension(filepath));

        tinyusdz::Stage stage;

        if (ext == ".usdc")
        {
            bool ret = tinyusdz::LoadUSDCFromFile(filepath.string(), &stage, &warn, &err);
            if (!warn.empty())
                pgLogWarn(warn);
            if (!err.empty())
                pgLogFatal(err);

            if (!ret) 
            {
                pgLogFatal("Failed to load USDC file:", filepath.string());
                return;
            }
        }
        else if (ext == ".usda")
        {
            bool ret = tinyusdz::LoadUSDAFromFile(filepath.string(), &stage, &warn, &err);
            if (!warn.empty())
                pgLogWarn(warn);
            if (!err.empty())
                pgLogFatal(err);

            if (!ret)
            {
                pgLogFatal("Failed to load USDA file:", filepath.string());
                return;
            }
        }
        else if (ext == ".usdz")
        {
            bool ret = tinyusdz::LoadUSDZFromFile(filepath.string(), &stage, &warn, &err);
            if (!warn.empty())
                pgLogWarn(warn);
            if (!err.empty())
                pgLogFatal(err);

            if (!ret)
            {
                pgLogFatal("Failed to load USDZ file:", filepath.string());
                return;
            }
        }
        else 
        {
            // Try to auto detect format
            bool ret = tinyusdz::LoadUSDFromFile(filepath.string(), &stage, &warn, &err);
            if (!warn.empty())
                pgLogWarn(warn);
            if (!err.empty())
                pgLogFatal(err);

            if (!ret)
            {
                pgLogFatal("Failed to load USD file:", filepath.string());
                return;
            }
        }

        auto primVisitFunc = [](const tinyusdz::Path& abs_path, const tinyusdz::Prim& prim, const int32_t level, void* userdata, std::string *err) -> bool {
            (void)err;
            std::cout << tinyusdz::pprint::Indent(level) << "[" << level << "] (" << prim.data().type_name() << ") " << prim.local_path().prim_part() << " : AbsPath " << tinyusdz::to_string(abs_path) << "\n";

            // Use as() or is() for Prim specific processing.
            if (const tinyusdz::Material *pm = prim.as<tinyusdz::Material>()) {
                (void)pm;
                std::cout << tinyusdz::pprint::Indent(level) << "  Got Material!\n";
                // return false + `err` empty if you want to terminate traversal earlier.
                //return false;
            }
            return true;
        };

        void* userdata = nullptr;

        tinyusdz::tydra::VisitPrims(stage, primVisitFunc, userdata);
    }

} // namespace prayground