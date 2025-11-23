#pragma once

#include <prayground/core/attribute.h>
#include <prayground/math/vec.h>
#include <prayground/shape/trianglemesh.h>
#include <prayground/ext/nanovdb/util/GridHandle.h>
#include <filesystem>
#include <vector>

namespace prayground {

    void loadObj(
        const std::filesystem::path& filepath, 
        std::vector<Vec3f>& vertices,
        std::vector<Face>& faces,
        std::vector<Vec3f>& normals,
        std::vector<Vec2f>& texcoords,
        bool trianglate = true
    );

    void loadObj(
        const std::filesystem::path& filepath, 
        TriangleMesh& mesh, 
        bool trianglate = true
    );

    void loadObjWithMtl(
        const std::filesystem::path& objpath, 
        std::vector<Vec3f>& vertices,
        std::vector<Face>& faces,
        std::vector<Vec3f>& normals,
        std::vector<Vec2f>& texcoords, 
        std::vector<uint32_t>& face_indices,
        std::vector<Attributes>& material_attribs,
        const std::filesystem::path& mtlpath,
        bool triangulate = true
    );

    void loadObjWithMtl(
        const std::filesystem::path& objpath, 
        const std::filesystem::path& mtlpath, 
        TriangleMesh& mesh, 
        std::vector<Attributes>& material_attribs,
        bool trianglate = true
    );

    // If .mtl file exists in same directory of .obj file
    void loadObjWithMtl(
        const std::filesystem::path& filepath, 
        TriangleMesh& mesh, 
        std::vector<Attributes>& material_attribs,
        bool trianglate = true
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

} // namespace prayground