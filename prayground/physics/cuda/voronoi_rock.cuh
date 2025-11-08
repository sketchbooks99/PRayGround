#pragma once

#include <prayground/physics/voronoi_rock.h>
#include <prayground/math/vec.h>

namespace prayground {

    // CUDA function to generate voronoi-based rock mesh
    extern "C" void buildVoronoiRockCUDA(
        const VoronoiRockParams& params,
        Vec3f** d_vertices_out,
        Vec3f** d_normals_out,
        Vec2f** d_texcoords_out,
        Vec3i** d_face_indices_out,
        Vec3i** d_normal_indices_out,
        Vec3i** d_texcoord_indices_out,
        int* vertex_count_out,
        int* face_count_out
    );

} // namespace prayground
