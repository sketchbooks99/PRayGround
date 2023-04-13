#pragma once

#include <prayground/core/attribute.h>
#include <prayground/math/vec.h>
#include <prayground/shape/trianglemesh.h>
#include <prayground/core/scene.h>
#include <prayground/core/file_util.h>
#include <nanovdb/util/GridHandle.h>
#include <filesystem>
#include <vector>


// TinyUSDZ
#include <tinyusdz/src/io-util.hh>
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

    void loadObj(
        const std::filesystem::path& filepath, 
        std::vector<Vec3f>& vertices,
        std::vector<Face>& faces,
        std::vector<Vec3f>& normals,
        std::vector<Vec2f>& texcoords
    );

    void loadObj(
        const std::filesystem::path& filepath, 
        TriangleMesh& mesh
    );

    void loadObjWithMtl(
        const std::filesystem::path& objpath, 
        std::vector<Vec3f>& vertices,
        std::vector<Face>& faces,
        std::vector<Vec3f>& normals,
        std::vector<Vec2f>& texcoords, 
        std::vector<uint32_t>& face_indices,
        std::vector<Attributes>& material_attribs,
        const std::filesystem::path& mtlpath
    );

    void loadObjWithMtl(
        const std::filesystem::path& objpath, 
        const std::filesystem::path& mtlpath, 
        TriangleMesh& mesh, 
        std::vector<Attributes>& material_attribs
    );

    // If .mtl file exists in same directory of .obj file
    void loadObjWithMtl(
        const std::filesystem::path& filepath, 
        TriangleMesh& mesh, 
        std::vector<Attributes>& material_attribs
    );

    void loadPly(
        const std::filesystem::path& filepath, 
        std::vector<Vec3f>& vertices,
        std::vector<Face>& faces, 
        std::vector<Vec3f>& normals,
        std::vector<Vec2f>& texcoords
    );

    // Load NanoVDB (not "OpenVDB" file!) 
    // This only accepts .nvdb file
    void loadNanoVDB(
        const std::filesystem::path& filepath, 
        nanovdb::GridHandle<>& handle
    );

    // Load USD file
    //template <DerivedFromCamera _CamT, uint32_t _NRay>
    inline bool loadUSDToScene(
        const std::filesystem::path& filepath,
        //Scene<_CamT, _NRay>& scene
        Scene<Camera, 1>& scene
    )
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
                return false;
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
                return false;
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
                return false;
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
                return false;
            }
        }

        return true;
    }

} // namespace prayground