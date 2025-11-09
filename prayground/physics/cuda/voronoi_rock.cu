#include <prayground/physics/voronoi_rock.h>
#include <prayground/physics/cuda/voronoi_rock.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace prayground {

    // Add vertex with atomic counter
    __device__ int addVertex(RockFieldBuffers* buffers, Vec3f pos, Vec3f normal, Vec2f texcoord) {
        int idx = atomicAdd(buffers->vertex_count, 1);
        if (idx < buffers->max_vertices) {
            buffers->vertices[idx] = pos;
            buffers->normals[idx] = normal;
            buffers->texcoords[idx] = texcoord;
        }
        return idx;
    }

    // Add face with atomic counter
    __device__ void addFace(RockFieldBuffers* buffers, Vec3i v_idx, Vec3i n_idx, Vec3i t_idx) {
        int idx = atomicAdd(buffers->face_count, 1);
        if (idx < buffers->max_faces) {
            buffers->face_indices[idx] = v_idx;
            buffers->normal_indices[idx] = n_idx;
            buffers->texcoord_indices[idx] = t_idx;
        }
    }

    // Initialize 2D Voronoi seeds on XZ plane
    __global__ void initializeVoronoiSeeds2D(
        VoronoiSeed2D* seeds,
        int seed_count,
        float field_size,
        uint32_t random_seed
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= seed_count) return;
        
        curandState state;
        curand_init(random_seed + idx, 0, 0, &state);
        
        // Random position in XZ plane
        float x = (curand_uniform(&state) - 0.5f) * field_size;
        float z = (curand_uniform(&state) - 0.5f) * field_size;
        seeds[idx].position = Vec2f(x, z);
        
        // Initialize cell size (will be computed later)
        seeds[idx].cell_size = 0.0f;
        seeds[idx].size_category = 0;
    }

    // Compute cell sizes (distance to nearest neighbor)
    __global__ void computeCellSizes(
        VoronoiSeed2D* seeds,
        int seed_count
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= seed_count) return;
        
        float min_dist = 1e10f;
        Vec2f my_pos = seeds[idx].position;
        
        // Find nearest neighbor (only if there are multiple seeds)
        if (seed_count > 1) {
            for (int i = 0; i < seed_count; i++) {
                if (i == idx) continue;
                float dist = length(seeds[i].position - my_pos);
                min_dist = fminf(min_dist, dist);
            }
        } else {
            // Single seed: use default cell size (full field)
            min_dist = 1.0f;  // Normalized default
        }
        
        seeds[idx].cell_size = min_dist;
    }

    // Categorize seeds by size (small, medium, large)
    __global__ void categorizeSeedSizes(
        VoronoiSeed2D* seeds,
        int seed_count,
        float* size_thresholds  // [small_threshold, large_threshold]
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= seed_count) return;
        
        float size = seeds[idx].cell_size;
        
        if (size < size_thresholds[0]) {
            seeds[idx].size_category = 0;  // Small
        } else if (size < size_thresholds[1]) {
            seeds[idx].size_category = 1;  // Medium
        } else {
            seeds[idx].size_category = 2;  // Large
        }
    }

    // Generate a single convex rock from random points
    __global__ void generateRocksKernel(
        VoronoiSeed2D* seeds,
        int seed_count,
        VoronoiRockParams params,
        RockFieldBuffers* buffers
    ) {
        int rock_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (rock_idx >= seed_count) return;
        
        VoronoiSeed2D& seed = seeds[rock_idx];
        
        // Initialize random state
        curandState state;
        curand_init(params.random_seed + rock_idx * 1000, 0, 0, &state);
        
        // Determine rock size based on rock_base_size (independent of field_size)
        // Use mix of absolute base size and relative cell size for variation
        float base_scale = params.rock_base_size;  // User-specified base size
        float relative_variation = seed.cell_size / params.field_size;  // 0-1 range
        
        float scale_multipliers[3] = {0.7f, 1.0f, 1.3f};  // Small, medium, large
        float rock_scale = base_scale * scale_multipliers[seed.size_category] * (0.8f + relative_variation * 0.4f);
        
        // Number of vertices for this rock
        int num_vertices = params.min_rock_vertices + 
            int(curand_uniform(&state) * (params.max_rock_vertices - params.min_rock_vertices));
        
        // Generate random points on a sphere (will form convex hull)
        Vec3f rock_vertices[64];  // Max 64 vertices per rock (stack allocation)
        if (num_vertices > 64) num_vertices = 64;
        
        float min_y = 1e10f;  // Track minimum Y coordinate
        
        for (int i = 0; i < num_vertices; i++) {
            // Random point on unit sphere using rejection sampling
            Vec3f p;
            float p_len;
            do {
                p.x() = curand_uniform(&state) * 2.0f - 1.0f;
                p.y() = curand_uniform(&state) * 2.0f - 1.0f;
                p.z() = curand_uniform(&state) * 2.0f - 1.0f;
                p_len = length(p);
            } while (p_len > 1.0f || p_len < 0.1f);
            
            p = p / p_len;  // Safe normalize (already checked p_len >= 0.1)
            
            // Add roughness (random radius variation)
            float radius = 1.0f + (curand_uniform(&state) - 0.5f) * params.rock_roughness;
            
            // Apply anisotropic scaling (squash Y slightly)
            float y_scale = 0.6f + curand_uniform(&state) * params.rock_height_variation;
            
            rock_vertices[i] = Vec3f(
                p.x() * radius * rock_scale,
                p.y() * radius * rock_scale * y_scale,
                p.z() * radius * rock_scale
            );
            
            // Track minimum Y for normalization
            if (rock_vertices[i].y() < min_y) {
                min_y = rock_vertices[i].y();
            }
            
            // Safety check for NaN in generated vertex
            if (!rock_vertices[i].isValid()) {
                // Force to small valid value
                rock_vertices[i] = Vec3f(0.1f, 0.1f, 0.1f);
            }
        }
        
        // Normalize Y coordinates so bottom is at Y=0
        for (int i = 0; i < num_vertices; i++) {
            rock_vertices[i].y() -= min_y;
        }
        
        // Rock center position
        // For single rock generation (seed_count=1), place at origin
        // For multi-rock generation, use seed position for distribution
        Vec3f rock_center;
        if (seed_count == 1) {
            // Single rock: always at origin (0, y_position, 0)
            rock_center = Vec3f(0.0f, params.y_position, 0.0f);
        } else {
            // Multiple rocks: XZ plane mapping from seed position
            rock_center = Vec3f(seed.position.x(), params.y_position, seed.position.y());
        }
        
        // Debug: Print first few rocks
        if (rock_idx < 5) {
            printf("[Rock %d] Position: (%.2f, %.2f, %.2f), Cell size: %.2f, Scale: %.2f, Vertices: %d, Category: %d, min_y_before_norm: %.2f\n",
                   rock_idx, rock_center.x(), rock_center.y(), rock_center.z(),
                   seed.cell_size, rock_scale, num_vertices, seed.size_category, min_y);
        }
        
        // Simple convex hull approximation: create triangles from centroid to all vertex pairs
        // This creates a star-shaped polyhedron (not perfect convex hull, but good enough)
        
        // Add centroid
        Vec3f centroid(0, 0, 0);
        for (int i = 0; i < num_vertices; i++) {
            centroid = centroid + rock_vertices[i];
        }
        centroid = centroid / float(num_vertices);
        
        // Sort vertices by Y coordinate first (latitude), then by angle (longitude)
        // This creates a more uniform triangulation around the sphere
        for (int i = 0; i < num_vertices - 1; i++) {
            for (int j = i + 1; j < num_vertices; j++) {
                Vec3f vi = rock_vertices[i];
                Vec3f vj = rock_vertices[j];
                
                // Primary sort by Y (height)
                if (vi.y() > vj.y() + 0.01f) {  // Small epsilon for stability
                    Vec3f temp = rock_vertices[i];
                    rock_vertices[i] = rock_vertices[j];
                    rock_vertices[j] = temp;
                } else if (fabsf(vi.y() - vj.y()) < 0.01f) {
                    // Secondary sort by angle in XZ plane
                    float angle_i = atan2f(vi.z(), vi.x());
                    float angle_j = atan2f(vj.z(), vj.x());
                    
                    if (angle_i > angle_j) {
                        Vec3f temp = rock_vertices[i];
                        rock_vertices[i] = rock_vertices[j];
                        rock_vertices[j] = temp;
                    }
                }
            }
        }
        
        // Create convex hull using simplified incremental algorithm
        // For small vertex counts (20-40), we can afford O(n^3) brute force
        int base_vertex_idx = atomicAdd(buffers->vertex_count, num_vertices);
        
        if (base_vertex_idx + num_vertices - 1 < buffers->max_vertices) {
            // Compute rock center (centroid in local space, already calculated above)
            // We'll use it for spherical UV mapping
            
            // Add all rock vertices to buffer with spherical UV mapping
            for (int i = 0; i < num_vertices; i++) {
                Vec3f world_pos = rock_center + rock_vertices[i];
                
                // Compute vertex normal: direction from origin to vertex
                // Since rock_vertices are already centered around origin,
                // this gives the correct outward normal for the convex hull
                Vec3f vertex_dir = rock_vertices[i];
                float vertex_len = length(vertex_dir);
                
                // Safety check: prevent NaN from zero-length vectors
                Vec3f normal;
                if (vertex_len > 1e-6f) {
                    normal = vertex_dir / vertex_len;
                } else {
                    // Fallback to upward normal if vertex is at origin (shouldn't happen)
                    normal = Vec3f(0.0f, 1.0f, 0.0f);
                }
                
                // Additional safety: check for NaN in normal
                if (isnan(normal.x()) || isnan(normal.y()) || isnan(normal.z())) {
                    normal = Vec3f(0.0f, 1.0f, 0.0f);
                }
                
                // Spherical UV mapping based on direction from rock center
                // N is already the normalized direction (normal)
                // Latitude φ = asin(N_y) ∈ [-π/2, π/2]
                // Longitude θ = atan2(N_x, N_z) ∈ [-π, π]
                float phi = asinf(clamp(normal.y(), -1.0f, 1.0f));  // Latitude
                float theta = atan2f(normal.x(), normal.z());        // Longitude
                
                // Normalize to [0, 1]
                // U = (θ + π) / (2π)  -> maps [-π, π] to [0, 1]
                // V = 1.0 - (φ + π/2) / π  -> maps [-π/2, π/2] to [1, 0] (top at V=0)
                float u = (theta + M_PI) / (2.0f * M_PI);
                float v = 1.0f - (phi + M_PI * 0.5f) / M_PI;
                
                buffers->vertices[base_vertex_idx + i] = world_pos;
                buffers->normals[base_vertex_idx + i] = normal;
                buffers->texcoords[base_vertex_idx + i] = Vec2f(u, v);
            }
            
            // Brute force convex hull: try all triangle combinations
            // For each triangle, check if all other points are on ONE side
            for (int i = 0; i < num_vertices; i++) {
                for (int j = i + 1; j < num_vertices; j++) {
                    for (int k = j + 1; k < num_vertices; k++) {
                        Vec3f v0 = rock_vertices[i];
                        Vec3f v1 = rock_vertices[j];
                        Vec3f v2 = rock_vertices[k];
                        
                        // Compute triangle normal (v0 → v1 → v2)
                        Vec3f edge1 = v1 - v0;
                        Vec3f edge2 = v2 - v0;
                        Vec3f face_normal = cross(edge1, edge2);
                        
                        // Skip degenerate triangles
                        float normal_len = length(face_normal);
                        if (normal_len < 0.0001f) continue;
                        face_normal = face_normal / normal_len;
                        
                        // Check if all other points are on one side of this plane
                        bool all_negative = true;
                        bool all_positive = true;
                        
                        for (int m = 0; m < num_vertices; m++) {
                            if (m == i || m == j || m == k) continue;
                            
                            Vec3f to_point = rock_vertices[m] - v0;
                            float side = dot(to_point, face_normal);
                            
                            if (side > 0.001f) all_negative = false;
                            if (side < -0.001f) all_positive = false;
                        }
                        
                        // If all points are on one side, this is a convex hull face
                        int idx0 = base_vertex_idx + i;
                        int idx1 = base_vertex_idx + j;
                        int idx2 = base_vertex_idx + k;
                        
                        // UV coordinates are already set via spherical mapping
                        // No need to recompute per-triangle
                        
                        if (all_positive) {
                            // All points on positive side → face_normal points outward
                            // Use CCW winding: v0 → v1 → v2
                            addFace(buffers,
                                Vec3i(idx0, idx1, idx2),
                                Vec3i(idx0, idx1, idx2),
                                Vec3i(idx0, idx1, idx2));
                        } else if (all_negative) {
                            // All points on negative side → face_normal points inward
                            // Flip to CCW: v0 → v2 → v1 (reverse order)
                            addFace(buffers,
                                Vec3i(idx0, idx2, idx1),
                                Vec3i(idx0, idx2, idx1),
                                Vec3i(idx0, idx2, idx1));
                        }
                    }
                }
            }
        }
    }

    // Host-side entry point
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
    ) {
        // Allocate 2D Voronoi seeds
        VoronoiSeed2D* d_seeds;
        cudaMalloc(&d_seeds, params.seed_count * sizeof(VoronoiSeed2D));
        
        int block_size = 256;
        int grid_size = (params.seed_count + block_size - 1) / block_size;
        
        // Initialize seeds
        initializeVoronoiSeeds2D<<<grid_size, block_size>>>(
            d_seeds, params.seed_count, params.field_size, params.random_seed
        );
        cudaDeviceSynchronize();
        
        // Compute cell sizes
        computeCellSizes<<<grid_size, block_size>>>(d_seeds, params.seed_count);
        cudaDeviceSynchronize();
        
        // Copy seeds to host to find size thresholds
        VoronoiSeed2D* h_seeds = new VoronoiSeed2D[params.seed_count];
        cudaMemcpy(h_seeds, d_seeds, params.seed_count * sizeof(VoronoiSeed2D), cudaMemcpyDeviceToHost);
        
        // Find 33rd and 66th percentile for size categories
        float* sizes = new float[params.seed_count];
        for (int i = 0; i < params.seed_count; i++) {
            sizes[i] = h_seeds[i].cell_size;
        }
        
        // Simple bubble sort (small array, no need for fancy sort)
        for (int i = 0; i < params.seed_count - 1; i++) {
            for (int j = i + 1; j < params.seed_count; j++) {
                if (sizes[i] > sizes[j]) {
                    float temp = sizes[i];
                    sizes[i] = sizes[j];
                    sizes[j] = temp;
                }
            }
        }
        
        float size_thresholds[2];
        size_thresholds[0] = sizes[params.seed_count / 3];      // 33rd percentile
        size_thresholds[1] = sizes[2 * params.seed_count / 3];  // 66th percentile
        
        delete[] h_seeds;
        delete[] sizes;
        
        float* d_thresholds;
        cudaMalloc(&d_thresholds, 2 * sizeof(float));
        cudaMemcpy(d_thresholds, size_thresholds, 2 * sizeof(float), cudaMemcpyHostToDevice);
        
        // Categorize seed sizes
        categorizeSeedSizes<<<grid_size, block_size>>>(d_seeds, params.seed_count, d_thresholds);
        cudaDeviceSynchronize();
        
        // Estimate maximum mesh size
        // Brute force convex hull can generate many duplicate faces before filtering
        // Each rock: ~40 vertices, theoretical max ~76 faces, but we generate C(n,3) candidates
        int max_vertices = params.seed_count * params.max_rock_vertices * 2;
        int max_faces = params.seed_count * params.max_rock_vertices * params.max_rock_vertices;  // Much larger buffer
        
        // Allocate mesh buffers
        RockFieldBuffers buffers;
        cudaMalloc(&buffers.vertices, max_vertices * sizeof(Vec3f));
        cudaMalloc(&buffers.normals, max_vertices * sizeof(Vec3f));
        cudaMalloc(&buffers.texcoords, max_vertices * sizeof(Vec2f));
        cudaMalloc(&buffers.face_indices, max_faces * sizeof(Vec3i));
        cudaMalloc(&buffers.normal_indices, max_faces * sizeof(Vec3i));
        cudaMalloc(&buffers.texcoord_indices, max_faces * sizeof(Vec3i));
        
        cudaMalloc(&buffers.vertex_count, sizeof(int));
        cudaMalloc(&buffers.face_count, sizeof(int));
        cudaMemset(buffers.vertex_count, 0, sizeof(int));
        cudaMemset(buffers.face_count, 0, sizeof(int));
        
        buffers.max_vertices = max_vertices;
        buffers.max_faces = max_faces;
        
        // Transfer buffers to device
        RockFieldBuffers* d_buffers;
        cudaMalloc(&d_buffers, sizeof(RockFieldBuffers));
        cudaMemcpy(d_buffers, &buffers, sizeof(RockFieldBuffers), cudaMemcpyHostToDevice);
        
        // Generate rocks
        generateRocksKernel<<<grid_size, block_size>>>(
            d_seeds, params.seed_count, params, d_buffers
        );
        cudaDeviceSynchronize();
        
        // Retrieve results
        int h_vertex_count, h_face_count;
        cudaMemcpy(&h_vertex_count, buffers.vertex_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_face_count, buffers.face_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        *vertex_count_out = h_vertex_count;
        *face_count_out = h_face_count;
        
        // Set output pointers
        *d_vertices_out = buffers.vertices;
        *d_normals_out = buffers.normals;
        *d_texcoords_out = buffers.texcoords;
        *d_face_indices_out = buffers.face_indices;
        *d_normal_indices_out = buffers.normal_indices;
        *d_texcoord_indices_out = buffers.texcoord_indices;
        
        // Cleanup temporary buffers
        cudaFree(d_seeds);
        cudaFree(d_thresholds);
        cudaFree(d_buffers);
        cudaFree(buffers.vertex_count);
        cudaFree(buffers.face_count);
    }

} // namespace prayground
