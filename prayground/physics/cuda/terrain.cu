#include <prayground/physics/terrain.h>
#include <prayground/physics/cuda/terrain.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>

namespace prayground {

// ============================================================================
// Kernel 1: Initialize heightmap with Perlin noise
// ============================================================================

__global__ void generateInitialHeightmapKernel(
    float* heightmap,
    int width,
    int height,
    float height_scale,
    int octaves,
    float frequency,
    float lacunarity,
    float persistence,
    unsigned int seed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || z >= height)
        return;
    
    // Normalize coordinates to [0, 1]
    float nx = (float)x / (width - 1);
    float nz = (float)z / (height - 1);
    
    // Generate multi-octave Perlin noise
    float noise = perlinFBM(nx, nz, octaves, frequency, lacunarity, persistence, seed);
    
    // Map from [-1, 1] to [0, height_scale]
    float h = (noise * 0.5f + 0.5f) * height_scale;

    heightmap[z * width + x] = h;
}

// ============================================================================
// Kernel 2: Simulate water droplet erosion
// ============================================================================

__global__ void simulateErosionKernel(
    float* heightmap,
    int width,
    int height,
    float terrain_size,
    int droplet_count,
    int max_lifetime,
    float inertia,
    float sediment_capacity_factor,
    float min_sediment_capacity,
    float erode_speed,
    float deposit_speed,
    float evaporate_speed,
    float gravity,
    float start_speed,
    float start_water,
    int brush_radius,
    unsigned int seed
) {
    int droplet_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (droplet_idx >= droplet_count)
        return;
    
    // Initialize random state for this droplet
    curandState rand_state;
    curand_init(seed + droplet_idx, 0, 0, &rand_state);
    
    // Random starting position (with margin from edges)
    const int edge_margin = 2;
    float pos_x = edge_margin + curand_uniform(&rand_state) * (width - 1 - 2 * edge_margin);
    float pos_z = edge_margin + curand_uniform(&rand_state) * (height - 1 - 2 * edge_margin);
    
    float dir_x = 0.0f;
    float dir_z = 0.0f;
    float speed = start_speed;
    float water = start_water;
    float sediment = 0.0f;
    
    float cell_size = terrain_size / (width - 1);
    
    for (int lifetime = 0; lifetime < max_lifetime; lifetime++) {
        int node_x = (int)pos_x;
        int node_z = (int)pos_z;
        
        // Check bounds with margin (prevent edge erosion artifacts)
        const int edge_margin = 2;
        if (node_x < edge_margin || node_x >= width - 1 - edge_margin || 
            node_z < edge_margin || node_z >= height - 1 - edge_margin)
            break;
        
        // Calculate droplet's offset inside the cell
        float cell_offset_x = pos_x - node_x;
        float cell_offset_z = pos_z - node_z;
        
        // Calculate droplet's height using bilinear interpolation
        float height_nw = getHeight(heightmap, node_x, node_z, width, height);
        float height_ne = getHeight(heightmap, node_x + 1, node_z, width, height);
        float height_sw = getHeight(heightmap, node_x, node_z + 1, width, height);
        float height_se = getHeight(heightmap, node_x + 1, node_z + 1, width, height);
        
        float height = bilerp(height_nw, height_ne, height_sw, height_se, 
                              cell_offset_x, cell_offset_z);
        
        // Calculate gradient
        Vec2f gradient = calculateGradient(heightmap, pos_x, pos_z, width, height, cell_size);
        
        // Update direction (mix of gradient and previous direction)
        dir_x = dir_x * inertia - gradient.x() * (1.0f - inertia);
        dir_z = dir_z * inertia - gradient.y() * (1.0f - inertia);
        
        // Normalize direction
        float len = sqrtf(dir_x * dir_x + dir_z * dir_z);
        if (len > 1e-6f) {
            dir_x /= len;
            dir_z /= len;
        } else {
            // If stuck, pick random direction
            float angle = curand_uniform(&rand_state) * 2.0f * M_PI;
            dir_x = cosf(angle);
            dir_z = sinf(angle);
        }
        
        // Move droplet
        float new_pos_x = pos_x + dir_x;
        float new_pos_z = pos_z + dir_z;
        
        // Sample height at new position
        float new_height = sampleHeight(heightmap, new_pos_x, new_pos_z, width, height);
        
        // Calculate height difference
        float delta_height = new_height - height;
        
        // Calculate sediment capacity
        float capacity = fmaxf(-delta_height, min_sediment_capacity) * 
                        speed * water * sediment_capacity_factor;
        
        // Erode or deposit sediment
        if (sediment > capacity || delta_height > 0.0f) {
            // Deposit sediment
            float amount_to_deposit = (delta_height > 0.0f) ? 
                                     fminf(delta_height, sediment) : 
                                     (sediment - capacity) * deposit_speed;
            
            sediment -= amount_to_deposit;
            
            // Calculate total brush weight to normalize
            float total_weight = 0.0f;
            for (int bz = -brush_radius; bz <= brush_radius; bz++) {
                for (int bx = -brush_radius; bx <= brush_radius; bx++) {
                    float dist = sqrtf(bx * bx + bz * bz);
                    if (dist <= brush_radius) {
                        total_weight += 1.0f - dist / brush_radius;
                    }
                }
            }
            
            // Deposit in a brush pattern around current position
            for (int bz = -brush_radius; bz <= brush_radius; bz++) {
                for (int bx = -brush_radius; bx <= brush_radius; bx++) {
                    int deposit_x = node_x + bx;
                    int deposit_z = node_z + bz;
                    
                    if (deposit_x >= 0 && deposit_x < width && 
                        deposit_z >= 0 && deposit_z < height) {
                        
                        float dist = sqrtf(bx * bx + bz * bz);
                        if (dist <= brush_radius) {
                            float weight = (1.0f - dist / brush_radius) / total_weight;
                            int idx = deposit_z * width + deposit_x;
                            atomicAdd(&heightmap[idx], amount_to_deposit * weight);
                        }
                    }
                }
            }
        } else {
            // Erode material
            // Amount to erode should be positive and limited by available material
            float amount_to_erode = fmaxf(0.0f, fminf((capacity - sediment) * erode_speed, -delta_height));
            
            // Calculate total brush weight to normalize
            float total_weight = 0.0f;
            for (int bz = -brush_radius; bz <= brush_radius; bz++) {
                for (int bx = -brush_radius; bx <= brush_radius; bx++) {
                    float dist = sqrtf(bx * bx + bz * bz);
                    if (dist <= brush_radius) {
                        total_weight += 1.0f - dist / brush_radius;
                    }
                }
            }
            
            // Erode in a brush pattern
            for (int bz = -brush_radius; bz <= brush_radius; bz++) {
                for (int bx = -brush_radius; bx <= brush_radius; bx++) {
                    int erode_x = node_x + bx;
                    int erode_z = node_z + bz;
                    
                    if (erode_x >= 0 && erode_x < width && 
                        erode_z >= 0 && erode_z < height) {
                        
                        float dist = sqrtf(bx * bx + bz * bz);
                        if (dist <= brush_radius) {
                            float weight = (1.0f - dist / brush_radius) / total_weight;
                            int idx = erode_z * width + erode_x;
                            atomicAdd(&heightmap[idx], -amount_to_erode * weight);
                        }
                    }
                }
            }
            
            sediment += amount_to_erode;
        }
        
        // Update speed and water
        speed = sqrtf(fmaxf(0.0f, speed * speed + delta_height * gravity));
        water *= (1.0f - evaporate_speed);
        
        // Stop if out of water
        if (water < 0.01f)
            break;
        
        // Update position
        pos_x = new_pos_x;
        pos_z = new_pos_z;
    }
}

// ============================================================================
// Kernel: Clamp heightmap to reasonable range
// ============================================================================

__global__ void clampHeightmapKernel(
    float* heightmap,
    int width,
    int height,
    float min_height,
    float max_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || z >= height)
        return;
    
    int idx = z * width + x;
    heightmap[idx] = fmaxf(min_height, fminf(max_height, heightmap[idx]));
}

// ============================================================================
// Kernel 3: Generate mesh vertices and normals
// ============================================================================

__global__ void generateMeshKernel(
    const float* heightmap,
    Vec3f* vertices,
    Vec3f* normals,
    Vec2f* texcoords,
    int width,
    int height,
    float terrain_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || z >= height)
        return;
    
    int idx = z * width + x;
    float cell_size = terrain_size / (width - 1);
    
    // Generate vertex position
    float world_x = (x - width / 2.0f) * cell_size;
    float world_z = (z - height / 2.0f) * cell_size;
    float world_y = heightmap[idx];
    
    vertices[idx] = Vec3f(world_x, world_y, world_z);
    
    // Generate UV coordinates
    texcoords[idx] = Vec2f((float)x / (width - 1), (float)z / (height - 1));
    
    // Calculate normal using finite differences
    float h_center = heightmap[idx];
    float h_left = (x > 0) ? heightmap[z * width + (x - 1)] : h_center;
    float h_right = (x < width - 1) ? heightmap[z * width + (x + 1)] : h_center;
    float h_down = (z > 0) ? heightmap[(z - 1) * width + x] : h_center;
    float h_up = (z < height - 1) ? heightmap[(z + 1) * width + x] : h_center;
    
    Vec3f tangent_x = Vec3f(2.0f * cell_size, h_right - h_left, 0.0f);
    Vec3f tangent_z = Vec3f(0.0f, h_up - h_down, 2.0f * cell_size);
    
    Vec3f normal = cross(tangent_z, tangent_x);
    normals[idx] = normalize(normal);
}

// ============================================================================
// Kernel 4: Generate triangle faces
// ============================================================================

__global__ void generateFacesKernel(
    uint3* faces,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width - 1 || z >= height - 1)
        return;
    
    int quad_idx = z * (width - 1) + x;
    int face_idx = quad_idx * 2;
    
    // Vertex indices for this quad
    unsigned int v00 = z * width + x;
    unsigned int v10 = z * width + (x + 1);
    unsigned int v01 = (z + 1) * width + x;
    unsigned int v11 = (z + 1) * width + (x + 1);
    
    // First triangle (counter-clockwise)
    faces[face_idx] = make_uint3(v00, v10, v01);
    
    // Second triangle (counter-clockwise)
    faces[face_idx + 1] = make_uint3(v01, v10, v11);
}

// ============================================================================
// Host function
// ============================================================================

void buildTerrainMeshCUDA(const TerrainParams& params, TerrainData& terrain_data) {
    printf("[Terrain] Starting generation...\n");
    printf("  Grid: %dx%d, Size: %.2f, Iterations: %d\n", 
           params.grid_width, params.grid_height, params.terrain_size, params.erosion_iterations);
    
    int width = params.grid_width;
    int height = params.grid_height;
    int num_vertices = width * height;
    int num_faces = 2 * (width - 1) * (height - 1);
    
    // Allocate device buffers
    float* d_heightmap;
    Vec3f* d_vertices;
    Vec3f* d_normals;
    Vec2f* d_texcoords;
    uint3* d_faces;
    
    cudaMalloc(&d_heightmap, num_vertices * sizeof(float));
    cudaMalloc(&d_vertices, num_vertices * sizeof(Vec3f));
    cudaMalloc(&d_normals, num_vertices * sizeof(Vec3f));
    cudaMalloc(&d_texcoords, num_vertices * sizeof(Vec2f));
    cudaMalloc(&d_faces, num_faces * sizeof(uint3));
    
    // Step 1: Generate initial heightmap with Perlin noise
    dim3 block_size(16, 16);
    dim3 grid_size((width + 15) / 16, (height + 15) / 16);
    
    generateInitialHeightmapKernel<<<grid_size, block_size>>>(
        d_heightmap, width, height,
        params.height_scale,
        params.noise_octaves,
        params.noise_frequency,
        params.noise_lacunarity,
        params.noise_persistence,
        params.noise_seed
    );
    cudaDeviceSynchronize();
    printf("[Terrain] Initial heightmap generated\n");
    
    // Step 2: Run erosion simulation
    int erosion_threads = 256;
    int erosion_blocks = (params.erosion_iterations + erosion_threads - 1) / erosion_threads;
    
    simulateErosionKernel<<<erosion_blocks, erosion_threads>>>(
        d_heightmap, width, height, params.terrain_size,
        params.erosion_iterations,
        params.max_droplet_lifetime,
        params.inertia,
        params.sediment_capacity_factor,
        params.min_sediment_capacity,
        params.erode_speed,
        params.deposit_speed,
        params.evaporate_speed,
        params.gravity,
        params.start_speed,
        params.start_water,
        params.erosion_brush_radius,
        params.noise_seed + 999
    );
    cudaDeviceSynchronize();
    printf("[Terrain] Erosion simulation completed (%d droplets)\n", params.erosion_iterations);
    
    // Step 2.5: Clamp heightmap to prevent extreme values
    clampHeightmapKernel<<<grid_size, block_size>>>(
        d_heightmap, width, height,
        -params.height_scale * 2.0f,  // Allow some erosion below 0
        params.height_scale * 2.0f    // Prevent extreme peaks
    );
    cudaDeviceSynchronize();
    printf("[Terrain] Heightmap clamped to valid range\n");
    
    // Step 3: Generate mesh vertices and normals
    generateMeshKernel<<<grid_size, block_size>>>(
        d_heightmap, d_vertices, d_normals, d_texcoords,
        width, height, params.terrain_size
    );
    cudaDeviceSynchronize();
    printf("[Terrain] Mesh vertices generated\n");
    
    // Step 4: Generate triangle faces
    generateFacesKernel<<<grid_size, block_size>>>(
        d_faces, width, height
    );
    cudaDeviceSynchronize();
    printf("[Terrain] Triangle faces generated\n");
    
    // Copy results back to host
    terrain_data.vertices.resize(num_vertices);
    terrain_data.normals.resize(num_vertices);
    terrain_data.texcoords.resize(num_vertices);
    terrain_data.faces.resize(num_faces);
    terrain_data.heightmap.resize(num_vertices);
    
    cudaMemcpy(terrain_data.vertices.data(), d_vertices, num_vertices * sizeof(Vec3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(terrain_data.normals.data(), d_normals, num_vertices * sizeof(Vec3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(terrain_data.texcoords.data(), d_texcoords, num_vertices * sizeof(Vec2f), cudaMemcpyDeviceToHost);
    cudaMemcpy(terrain_data.faces.data(), d_faces, num_faces * sizeof(uint3), cudaMemcpyDeviceToHost);
    cudaMemcpy(terrain_data.heightmap.data(), d_heightmap, num_vertices * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate min/max height for texture normalization
    float min_height = terrain_data.heightmap[0];
    float max_height = terrain_data.heightmap[0];
    for (int i = 1; i < num_vertices; i++) {
        float h = terrain_data.heightmap[i];
        if (h < min_height) min_height = h;
        if (h > max_height) max_height = h;
    }
    
    terrain_data.grid_width = width;
    terrain_data.grid_height = height;
    terrain_data.terrain_size = params.terrain_size;
    terrain_data.min_height = min_height;
    terrain_data.max_height = max_height;
    
    printf("[Terrain] Heightmap range: %.2f to %.2f\n", min_height, max_height);
    
    // Free device memory
    cudaFree(d_heightmap);
    cudaFree(d_vertices);
    cudaFree(d_normals);
    cudaFree(d_texcoords);
    cudaFree(d_faces);
    
    printf("[Terrain] Generated: %d vertices, %d faces\n", num_vertices, num_faces);
}

} // namespace prayground
