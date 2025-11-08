#pragma once

#include <prayground/physics/tree.h>
#include <prayground/math/vec.h>

namespace prayground {

    // Leaf generation parameters
    struct LeafGenerationParams {
        float leaf_density;
        int leaf_start_gen;
        float leaf_size;
        uint32_t seed;
        int num_leaf_textures;
    };

    // CUDA function declarations
    extern "C" void buildTreeMeshCUDA(
        CudaTreeBranch* d_branches,
        CudaTreeSegment* d_segments,
        int branch_count,
        int segment_count,
        int radial_segments,
        Vec3f** d_vertices_out,
        Vec3f** d_normals_out,
        Vec2f** d_texcoords_out,
        Vec3i** d_face_indices_out,
        Vec3i** d_normal_indices_out,
        Vec3i** d_texcoord_indices_out,
        int* vertex_count_out,
        int* face_count_out
    );

    extern "C" void buildLeafMeshCUDA(
        CudaTreeBranch* d_branches,
        CudaTreeSegment* d_segments,
        int branch_count,
        int segment_count,
        LeafGenerationParams params,
        Vec3f** d_leaf_vertices_out,
        Vec3f** d_leaf_normals_out,
        Vec2f** d_leaf_texcoords_out,
        Vec3i** d_leaf_face_indices_out,
        Vec3i** d_leaf_normal_indices_out,
        Vec3i** d_leaf_texcoord_indices_out,
        uint32_t** d_leaf_sbt_indices_out,
        int* leaf_vertex_count_out,
        int* leaf_face_count_out
    );

} // namespace prayground

