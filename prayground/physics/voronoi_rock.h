#pragma once

#include <prayground/math/vec.h>
#include <vector>
#include <memory>

namespace prayground {

    // Voronoi rock field generation parameters
    struct VoronoiRockParams {
        int seed_count;              // Number of Voronoi seed points (= number of rocks)
        float field_size;            // XZ field size (square area)
        float y_position;            // Y position for rock placement
        int min_rock_vertices;       // Minimum vertices per rock
        int max_rock_vertices;       // Maximum vertices per rock
        float rock_height_variation; // Height variation (0-1)
        float rock_roughness;        // Surface roughness (0-1)
        uint32_t random_seed;        // Random seed for reproducibility
    };

    // 2D Voronoi seed point for rock placement
    struct VoronoiSeed2D {
        Vec2f position;              // XZ position
        float cell_size;             // Estimated cell size (distance to nearest neighbor)
        int size_category;           // 0=small, 1=medium, 2=large
    };

    // Single rock mesh data
    struct RockMeshData {
        int vertex_start;
        int vertex_count;
        int face_start;
        int face_count;
        Vec3f position;              // Rock center position
        float scale;                 // Rock scale factor
    };

    // Rock field mesh buffers (all rocks combined)
    struct RockFieldBuffers {
        Vec3f* vertices;
        Vec3f* normals;
        Vec2f* texcoords;
        Vec3i* face_indices;
        Vec3i* normal_indices;
        Vec3i* texcoord_indices;
        
        int* vertex_count;
        int* face_count;
        int max_vertices;
        int max_faces;
    };

    // Default parameters for rock field
    inline VoronoiRockParams createDefaultRockFieldParams() {
        VoronoiRockParams params;
        params.seed_count = 30;
        params.field_size = 100.0f;
        params.y_position = 0.0f;
        params.min_rock_vertices = 20;
        params.max_rock_vertices = 40;
        params.rock_height_variation = 0.3f;
        params.rock_roughness = 0.2f;
        params.random_seed = 12345;
        return params;
    }

} // namespace prayground
