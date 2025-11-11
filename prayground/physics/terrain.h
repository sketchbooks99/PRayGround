#pragma once

#include <prayground/math/vec.h>
#include <vector>

namespace prayground {

// Terrain generation parameters
struct TerrainParams {
    // Grid settings
    int grid_width = 512;           // Number of vertices in X direction
    int grid_height = 512;          // Number of vertices in Z direction
    float terrain_size = 100.0f;    // Physical size of terrain (world units)

    // Initial height generation (Perlin noise)
    float height_scale = 10.0f;     // Maximum height variation
    int noise_octaves = 6;          // Number of noise layers
    float noise_frequency = 2.0f;    // Base frequency of noise
    float noise_lacunarity = 2.0f;   // Frequency multiplier per octave
    float noise_persistence = 0.5f;  // Amplitude multiplier per octave
    unsigned int noise_seed = 1234;     // Random seed for reproducibility

    // Erosion simulation parameters
    int erosion_iterations = 100000;   // Number of water droplets to simulate
    int max_droplet_lifetime = 30; // Maximum steps per droplet
    float inertia = 0.05f;           // Droplet inertia (0-1): higher = straighter paths
    float sediment_capacity_factor = 4.0f; // Multiplier for sediment carrying capacity
    float min_sediment_capacity = 0.01f;    // Minimum sediment capacity
    float erode_speed = 0.3f;       // Speed at which material is eroded
    float deposit_speed = 0.3f;     // Speed at which material is deposited
    float evaporate_speed = 0.1f;   // Rate of water evaporation (0-1)
    float gravity = 9.81f;          // Downward acceleration
    float start_speed = 1.0f;       // Initial droplet speed
    float start_water = 1.0f;       // Initial water volume

    // Erosion brush settings
    int erosion_brush_radius = 3; // Radius of erosion effect (in grid cells)
};

// Buffers for terrain mesh data
struct TerrainMeshBuffers {
    // Mesh data
    Vec3f* vertices;         // Vertex positions (grid_width * grid_height)
    Vec3f* normals;          // Vertex normals
    Vec2f* texcoords;        // UV coordinates
    uint3* faces;            // Triangle faces (2 * (grid_width-1) * (grid_height-1))
    
    // Buffer sizes
    int num_vertices;
    int num_faces;
    
    // Height map (used during erosion simulation)
    float* heightmap;        // Height values at each grid point
};

// Host-side terrain data container
struct TerrainData {
    std::vector<Vec3f> vertices;
    std::vector<Vec3f> normals;
    std::vector<Vec2f> texcoords;
    std::vector<uint3> faces;
    std::vector<float> heightmap;  // CPU-side heightmap for texture generation
    
    int grid_width;
    int grid_height;
    float terrain_size;
    float min_height;  // Minimum height value in heightmap
    float max_height;  // Maximum height value in heightmap
};

// CUDA function declaration
void buildTerrainMeshCUDA(
    const TerrainParams& params,
    TerrainData& terrain_data
);

} // namespace prayground
