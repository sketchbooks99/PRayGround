#include <cassert>
#include <vector>
#include <cstdio>
#include <prayground/physics/tree.h>
#include <prayground/physics/cuda/tree.cuh>
#include <prayground/core/onb.h>

namespace prayground {

    // Add vertex using atomic counter
    __device__ int addVertex(TreeMeshBuffers* buffers, Vec3f pos, Vec3f normal, Vec2f texcoord) {
        int idx = atomicAdd(buffers->vertex_count, 1);
        if (idx < buffers->max_vertices) {
            buffers->vertices[idx] = pos;
            buffers->normals[idx] = normal;
            buffers->texcoords[idx] = texcoord;
        }
        return idx;
    }

    // Add face using atomic counter
    __device__ void addFace(TreeMeshBuffers* buffers, Vec3i v_idx, Vec3i n_idx, Vec3i t_idx) {
        int idx = atomicAdd(buffers->face_count, 1);
        if (idx < buffers->max_faces) {
            buffers->face_indices[idx] = v_idx;
            buffers->normal_indices[idx] = n_idx;
            buffers->texcoord_indices[idx] = t_idx;
        }
    }

    // Find or create vertex for vertex sharing (simple version: linear search within tolerance)
    __device__ int findOrCreateVertex(TreeMeshBuffers* buffers, Vec3f pos, Vec3f normal, Vec2f texcoord, float tolerance = 0.5f) {
        int count = *buffers->vertex_count;

        // Linear search for existing vertices with tolerance for branch connections
        // Larger default tolerance helps connect branches smoothly at junction points
        for (int i = 0; i < count; i++) {
            Vec3f existing_pos = buffers->vertices[i];
            float dist = length(existing_pos - pos);

            if (dist < tolerance) {
                // Found existing vertex: average normals for smooth shading across branches
                Vec3f existing_normal = buffers->normals[i];
                Vec3f averaged = existing_normal + normal;
                float avg_len = length(averaged);
                
                // Safety: prevent NaN from canceling normals
                if (avg_len > 1e-6f) {
                    buffers->normals[i] = averaged / avg_len;
                } else {
                    // Keep existing normal if they cancel out
                    buffers->normals[i] = existing_normal;
                }
                return i;
            }
        }

        // Add new vertex
        return addVertex(buffers, pos, normal, texcoord);
    }

    // Kernel to meshify branches of a specific generation
    __global__ void meshifyBranchByGenerationKernel(
        CudaTreeBranch* branches,
        CudaTreeSegment* segments,
        int branch_count,
        int target_generation,
        int radial_segments,
        TreeMeshBuffers* buffers
    ) {
        int branch_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (branch_idx >= branch_count) return;

        CudaTreeBranch& branch = branches[branch_idx];
        
        // Only process branches of target generation
        if (branch.generation != target_generation) return;
        
        if (branch.segment_count < 2) return;

        // Get base segment orientation
        CudaTreeSegment& base_seg = segments[branch.segment_start];
        Vec3f tangent = base_seg.normal;
        Vec3f binormal = base_seg.binormal;

        // Previous segment's end_ring (temp buffer)
        int prev_end_ring[32];  // Assume max 32 divisions (stack memory)
        if (radial_segments > 32) return;  // Safety check

        // Create cylinder between each segment
        for (int i = 0; i < branch.segment_count - 1; i++) {
            CudaTreeSegment& start_seg = segments[branch.segment_start + i];
            CudaTreeSegment& end_seg = segments[branch.segment_start + i + 1];

            int start_ring[32];
            int end_ring[32];

            // Create start_ring
            for (int j = 0; j < radial_segments; j++) {
                float angle = 2.0f * math::pi * j / radial_segments;
                Vec3f offset = start_seg.radius * (tangent * cosf(angle) + binormal * sinf(angle));
                Vec3f pos = start_seg.position + offset;
                
                // Safe normalize for normal
                float offset_len = length(offset);
                Vec3f normal = (offset_len > 1e-6f) ? (offset / offset_len) : Vec3f(0.0f, 1.0f, 0.0f);
                
                Vec2f texcoord = Vec2f((float)j / radial_segments, (float)i / (branch.segment_count - 1));

                if (i == 0) {
                    // First segment - search for existing vertices from parent (parent was processed in earlier generation)
                    float connection_tolerance = start_seg.radius * 0.8f;  // Large tolerance for branch connections
                    start_ring[j] = findOrCreateVertex(buffers, pos, normal, texcoord, connection_tolerance);
                } else {
                    // Subsequent segments - ALWAYS reuse previous end_ring
                    start_ring[j] = prev_end_ring[j];
                }
            }

            // Create end_ring
            for (int j = 0; j < radial_segments; j++) {
                float angle = 2.0f * math::pi * j / radial_segments;
                Vec3f offset = end_seg.radius * (tangent * cosf(angle) + binormal * sinf(angle));
                Vec3f pos = end_seg.position + offset;
                
                // Safe normalize for normal
                float offset_len = length(offset);
                Vec3f normal = (offset_len > 1e-6f) ? (offset / offset_len) : Vec3f(0.0f, 1.0f, 0.0f);
                
                Vec2f texcoord = Vec2f((float)j / radial_segments, (float)(i + 1) / (branch.segment_count - 1));

                // Always create vertices for end_ring
                end_ring[j] = findOrCreateVertex(buffers, pos, normal, texcoord, 0.01f);
            }

            // Create side faces
            for (int j = 0; j < radial_segments; j++) {
                int next_j = (j + 1) % radial_segments;

                addFace(buffers,
                    Vec3i(start_ring[j], end_ring[j], start_ring[next_j]),
                    Vec3i(start_ring[j], end_ring[j], start_ring[next_j]),
                    Vec3i(start_ring[j], end_ring[j], start_ring[next_j]));

                addFace(buffers,
                    Vec3i(start_ring[next_j], end_ring[j], end_ring[next_j]),
                    Vec3i(start_ring[next_j], end_ring[j], end_ring[next_j]),
                    Vec3i(start_ring[next_j], end_ring[j], end_ring[next_j]));
            }

            for (int j = 0; j < radial_segments; j++) {
                prev_end_ring[j] = end_ring[j];
            }

            if (i == branch.segment_count - 2 && branch.is_terminal) {
                CudaTreeSegment& last_seg = segments[branch.segment_start + branch.segment_count - 1];
                Vec3f cap_center = last_seg.position;
                Vec3f cap_normal = last_seg.direction;
                int cap_center_idx = findOrCreateVertex(buffers, cap_center, cap_normal, Vec2f(0.5f, 1.0f), 0.01f);

                for (int j = 0; j < radial_segments; j++) {
                    int next_j = (j + 1) % radial_segments;
                    addFace(buffers,
                        Vec3i(cap_center_idx, end_ring[next_j], end_ring[j]),
                        Vec3i(cap_center_idx, end_ring[next_j], end_ring[j]),
                        Vec3i(cap_center_idx, end_ring[next_j], end_ring[j]));
                }
            }
        }
    }

    // Kernel to meshify branches of a specific generation
    __global__ void meshifyBranchKernel(
        CudaTreeBranch* branches,
        CudaTreeSegment* segments,
        int branch_count,
        int target_generation,  // Only process branches of this generation
        int radial_segments,
        TreeMeshBuffers* buffers
    ) {
        int branch_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (branch_idx >= branch_count) return;

        CudaTreeBranch& branch = branches[branch_idx];
        
        // Skip if not target generation
        if (branch.generation != target_generation) return;
        
        if (branch.segment_count < 2) return;

        // Get base segment orientation
        CudaTreeSegment& base_seg = segments[branch.segment_start];
        Vec3f tangent = base_seg.normal;
        Vec3f binormal = base_seg.binormal;

        // Previous segment's end_ring (temp buffer)
        int prev_end_ring[32];  // Assume max 32 divisions (stack memory)
        if (radial_segments > 32) return;  // Safety check

        // Create cylinder between each segment
        for (int i = 0; i < branch.segment_count - 1; i++) {
            CudaTreeSegment& start_seg = segments[branch.segment_start + i];
            CudaTreeSegment& end_seg = segments[branch.segment_start + i + 1];

            int start_ring[32];
            int end_ring[32];

            // Create start_ring
            for (int j = 0; j < radial_segments; j++) {
                float angle = 2.0f * math::pi * j / radial_segments;
                Vec3f offset = start_seg.radius * (tangent * cosf(angle) + binormal * sinf(angle));
                Vec3f pos = start_seg.position + offset;
                Vec3f normal = normalize(offset);
                Vec2f texcoord = Vec2f((float)j / radial_segments, (float)i / (branch.segment_count - 1));

                if (i == 0) {
                    // First segment - always try to reuse existing vertices from parent branch
                    // Use radius-based tolerance to ensure connection (parent and child have same radius at junction)
                    float connection_tolerance = start_seg.radius * 0.5f;  // 50% of radius should cover any angle differences
                    start_ring[j] = findOrCreateVertex(buffers, pos, normal, texcoord, connection_tolerance);
                } else {
                    // Subsequent segments - ALWAYS reuse previous end_ring (no gap within same branch)
                    start_ring[j] = prev_end_ring[j];
                }
            }

            // Create end_ring
            for (int j = 0; j < radial_segments; j++) {
                float angle = 2.0f * math::pi * j / radial_segments;
                Vec3f offset = end_seg.radius * (tangent * cosf(angle) + binormal * sinf(angle));
                Vec3f pos = end_seg.position + offset;
                Vec3f normal = normalize(offset);
                Vec2f texcoord = Vec2f((float)j / radial_segments, (float)(i + 1) / (branch.segment_count - 1));

                // Always try to reuse vertices - especially important at branch endpoints
                end_ring[j] = findOrCreateVertex(buffers, pos, normal, texcoord, 0.1f);
            }

            // Create side faces
            for (int j = 0; j < radial_segments; j++) {
                int next_j = (j + 1) % radial_segments;

                // Triangle 1
                addFace(buffers,
                    Vec3i(start_ring[j], end_ring[j], start_ring[next_j]),
                    Vec3i(start_ring[j], end_ring[j], start_ring[next_j]),
                    Vec3i(start_ring[j], end_ring[j], start_ring[next_j]));

                // Triangle 2
                addFace(buffers,
                    Vec3i(start_ring[next_j], end_ring[j], end_ring[next_j]),
                    Vec3i(start_ring[next_j], end_ring[j], end_ring[next_j]),
                    Vec3i(start_ring[next_j], end_ring[j], end_ring[next_j]));
            }

            // Save previous end_ring
            for (int j = 0; j < radial_segments; j++) {
                prev_end_ring[j] = end_ring[j];
            }

            // For last segment, add terminal cap if needed
            if (i == branch.segment_count - 2 && branch.is_terminal) {
                // Terminal branch - add cap
                CudaTreeSegment& last_seg = segments[branch.segment_start + branch.segment_count - 1];
                Vec3f cap_center = last_seg.position;
                Vec3f cap_normal = last_seg.direction;
                int cap_center_idx = findOrCreateVertex(buffers, cap_center, cap_normal, Vec2f(0.5f, 1.0f));

                for (int j = 0; j < radial_segments; j++) {
                    int next_j = (j + 1) % radial_segments;
                    addFace(buffers,
                        Vec3i(cap_center_idx, end_ring[next_j], end_ring[j]),
                        Vec3i(cap_center_idx, end_ring[next_j], end_ring[j]),
                        Vec3i(cap_center_idx, end_ring[next_j], end_ring[j]));
                }
            }
        }
    }

    // Host-side callable function
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
    ) {
        // Estimate maximum size
        int max_vertices = segment_count * radial_segments * 2;  // With margin
        int max_faces = segment_count * radial_segments * 4;

        // Allocate device memory
        TreeMeshBuffers buffers;
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
        TreeMeshBuffers* d_buffers;
        cudaMalloc(&d_buffers, sizeof(TreeMeshBuffers));
        cudaMemcpy(d_buffers, &buffers, sizeof(TreeMeshBuffers), cudaMemcpyHostToDevice);

        // Find maximum generation by copying branches to host
        std::vector<CudaTreeBranch> h_branches(branch_count);
        cudaMemcpy(h_branches.data(), d_branches, branch_count * sizeof(CudaTreeBranch), cudaMemcpyDeviceToHost);
        
        int max_generation = 1;
        for (int i = 0; i < branch_count; i++) {
            if (h_branches[i].generation > max_generation) {
                max_generation = h_branches[i].generation;
            }
        }

        printf("[Tree Mesh] Processing %d generations sequentially to avoid gaps\n", max_generation);

        // Launch kernel for each generation sequentially (parent before children)
        int block_size = 256;
        int grid_size = (branch_count + block_size - 1) / block_size;
        
        for (int gen = 1; gen <= max_generation; gen++) {
            meshifyBranchKernel<<<grid_size, block_size>>>(
                d_branches, d_segments,
                branch_count, gen, radial_segments, d_buffers
            );
            cudaDeviceSynchronize();  // Wait for this generation to complete before next
        }

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

        // Free temporary buffers
        cudaFree(d_buffers);
        cudaFree(buffers.vertex_count);
        cudaFree(buffers.face_count);
    }

    // ============================================================================
    // Leaf Generation
    // ============================================================================

    // Simple LCG random number generator for CUDA
    __device__ float cuda_rnd(uint32_t& seed) {
        seed = (1103515245U * seed + 12345U) & 0x7fffffffU;
        return static_cast<float>(seed) / static_cast<float>(0x7fffffffU);
    }

    // Kernel to generate leaves for all eligible branches
    __global__ void generateLeavesKernel(
        CudaTreeBranch* branches,
        CudaTreeSegment* segments,
        int branch_count,
        LeafGenerationParams params,
        TreeMeshBuffers* buffers
    ) {
        int branch_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (branch_idx >= branch_count) return;

        CudaTreeBranch& branch = branches[branch_idx];
        
        // Only generate leaves for branches at or beyond leaf_start_gen
        if (branch.generation < params.leaf_start_gen) return;

        int seg_start = branch.segment_start;
        int segs_in_branch = branch.segment_count;
        
        if (segs_in_branch < 2) return;  // Need at least 2 segments for interpolation

        // Calculate number of leaves for this branch
        // Terminal branches (tips) get more leaves
        float density_multiplier = branch.is_terminal ? 1.5f : 1.0f;
        int num_leaves = (int)(segs_in_branch * params.leaf_density * density_multiplier);
        
        // Unique seed per branch and leaf
        uint32_t base_seed = params.seed + branch_idx * 1000;

        for (int leaf_idx = 0; leaf_idx < num_leaves; leaf_idx++) {
            uint32_t seed = base_seed + leaf_idx;

            // Random position along branch (0.0 to 1.0)
            float branch_t = cuda_rnd(seed);
            
            // Map to segment range (allow full range including tip)
            float segment_float = branch_t * (float)(segs_in_branch - 1);
            int seg_local_idx = (int)segment_float;
            float seg_t = segment_float - (float)seg_local_idx;
            
            // Clamp to valid range for interpolation
            seg_local_idx = seg_local_idx < (segs_in_branch - 1) ? seg_local_idx : (segs_in_branch - 2);

            // Get two consecutive segments
            CudaTreeSegment& seg0 = segments[seg_start + seg_local_idx];
            CudaTreeSegment& seg1 = segments[seg_start + seg_local_idx + 1];

            // Interpolate segment properties
            Vec3f position = seg0.position * (1.0f - seg_t) + seg1.position * seg_t;
            Vec3f direction = normalize(seg0.direction * (1.0f - seg_t) + seg1.direction * seg_t);
            Vec3f normal = normalize(seg0.normal * (1.0f - seg_t) + seg1.normal * seg_t);
            Vec3f binormal = normalize(seg0.binormal * (1.0f - seg_t) + seg1.binormal * seg_t);
            float radius = seg0.radius * (1.0f - seg_t) + seg1.radius * seg_t;

            // Random rotation around branch (0 to 2*PI)
            float rotation_angle = cuda_rnd(seed) * 2.0f * math::pi;
            Vec3f leaf_normal = normal * cosf(rotation_angle) + binormal * sinf(rotation_angle);
            leaf_normal = normalize(leaf_normal);

            // Add droop angle - leaf droops down from branch
            float droop_angle = cuda_rnd(seed) * math::pi * 0.4f;  // 0 to 72 degrees droop
            Vec3f leaf_direction = leaf_normal * cosf(droop_angle) - direction * sinf(droop_angle);
            leaf_direction = normalize(leaf_direction);

            // Leaf base position on branch surface
            Vec3f branch_surface_pos = position + leaf_normal * radius;

            // Random leaf orientation - completely random direction for leaf surface
            // Generate random angles for spherical coordinates
            float random_theta = cuda_rnd(seed) * math::pi;  // 0 to PI
            float random_phi = cuda_rnd(seed) * 2.0f * math::pi;  // 0 to 2*PI
            
            // Random tangent direction
            Vec3f random_tangent = Vec3f(
                sinf(random_theta) * cosf(random_phi),
                sinf(random_theta) * sinf(random_phi),
                cosf(random_theta)
            );
            random_tangent = normalize(random_tangent);
            
            // Make sure tangent is perpendicular to leaf direction
            Vec3f leaf_tangent = normalize(random_tangent - leaf_direction * dot(random_tangent, leaf_direction));
            
            // Fallback if parallel
            if (dot(leaf_tangent, leaf_tangent) < 0.01f) {
                leaf_tangent = normalize(cross(leaf_direction, Vec3f(0, 1, 0)));
                if (dot(leaf_tangent, leaf_tangent) < 0.01f) {
                    leaf_tangent = normalize(cross(leaf_direction, Vec3f(1, 0, 0)));
                }
            }

            // Random size variation (80% to 120% of base size)
            float size_variation = 0.8f + cuda_rnd(seed) * 0.4f;
            float leaf_length = params.leaf_size * size_variation;
            float leaf_width = params.leaf_size * 0.6f * size_variation;

            // Create billboard quad (4 vertices, 2 triangles)
            Vec3f v0 = branch_surface_pos - leaf_tangent * leaf_width * 0.5f;
            Vec3f v1 = branch_surface_pos + leaf_tangent * leaf_width * 0.5f;
            Vec3f v2 = branch_surface_pos + leaf_tangent * leaf_width * 0.5f + leaf_direction * leaf_length;
            Vec3f v3 = branch_surface_pos - leaf_tangent * leaf_width * 0.5f + leaf_direction * leaf_length;

            // UV coordinates
            Vec2f uv0(1, 0);
            Vec2f uv1(1, 1);
            Vec2f uv2(0, 1);
            Vec2f uv3(0, 0);

            // Normal for all vertices - perpendicular to leaf surface
            Vec3f face_normal = normalize(cross(leaf_tangent, leaf_direction));

            // Add vertices
            int idx0 = addVertex(buffers, v0, face_normal, uv0);
            int idx1 = addVertex(buffers, v1, face_normal, uv1);
            int idx2 = addVertex(buffers, v2, face_normal, uv2);
            int idx3 = addVertex(buffers, v3, face_normal, uv3);

            // Add two triangles
            addFace(buffers, Vec3i(idx0, idx1, idx2), Vec3i(idx0, idx1, idx2), Vec3i(idx0, idx1, idx2));
            addFace(buffers, Vec3i(idx0, idx2, idx3), Vec3i(idx0, idx2, idx3), Vec3i(idx0, idx2, idx3));

            // Store SBT index (random texture selection)
            if (params.num_leaf_textures > 0) {
                int texture_idx = (int)(cuda_rnd(seed) * params.num_leaf_textures) % params.num_leaf_textures;
                // TODO: Store SBT index - need separate buffer
            }
        }
    }

    // Host function to build leaf mesh
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
    ) {
        // Estimate max leaves based on density
        // Each segment can have multiple leaves based on density
        // Conservative estimate: segments_per_branch * density * terminal_multiplier
        int avg_segments_per_branch = max(10, segment_count / max(1, branch_count));
        float max_density_with_multiplier = params.leaf_density * 1.5f;  // Terminal multiplier
        int max_leaves_per_branch = (int)(avg_segments_per_branch * max_density_with_multiplier) + 10;  // Add safety margin
        
        int max_leaves = branch_count * max_leaves_per_branch;
        int max_vertices = max_leaves * 4;   // 4 vertices per leaf
        int max_faces = max_leaves * 2;      // 2 triangles per leaf
        
        printf("[CUDA Leaf] Max estimate: %d leaves (%d per branch), %d vertices, %d faces\n",
               max_leaves, max_leaves_per_branch, max_vertices, max_faces);

        // Allocate buffers
        TreeMeshBuffers buffers;
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
        TreeMeshBuffers* d_buffers;
        cudaMalloc(&d_buffers, sizeof(TreeMeshBuffers));
        cudaMemcpy(d_buffers, &buffers, sizeof(TreeMeshBuffers), cudaMemcpyHostToDevice);

        // Launch kernel
        int block_size = 256;
        int grid_size = (branch_count + block_size - 1) / block_size;
        
        printf("[CUDA Leaf] Launching kernel: grid=%d, block=%d, branches=%d\n", 
               grid_size, block_size, branch_count);
        
        generateLeavesKernel<<<grid_size, block_size>>>(
            d_branches, d_segments,
            branch_count, params, d_buffers
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[CUDA Error] Kernel launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[CUDA Error] Kernel execution failed: %s\n", cudaGetErrorString(err));
            return;
        }

        // Retrieve results
        int h_vertex_count, h_face_count;
        cudaMemcpy(&h_vertex_count, buffers.vertex_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_face_count, buffers.face_count, sizeof(int), cudaMemcpyDeviceToHost);

        // Clamp to max size (防止溢出)
        if (h_vertex_count > max_vertices) {
            printf("[CUDA Leaf Warning] Vertex count %d exceeds max %d, clamping\n", h_vertex_count, max_vertices);
            h_vertex_count = max_vertices;
        }
        if (h_face_count > max_faces) {
            printf("[CUDA Leaf Warning] Face count %d exceeds max %d, clamping\n", h_face_count, max_faces);
            h_face_count = max_faces;
        }

        *leaf_vertex_count_out = h_vertex_count;
        *leaf_face_count_out = h_face_count;

        printf("[CUDA Leaf Gen] Generated %d vertices, %d faces (max: %d v, %d f)\n", 
               h_vertex_count, h_face_count, max_vertices, max_faces);

        // Set output pointers
        *d_leaf_vertices_out = buffers.vertices;
        *d_leaf_normals_out = buffers.normals;
        *d_leaf_texcoords_out = buffers.texcoords;
        *d_leaf_face_indices_out = buffers.face_indices;
        *d_leaf_normal_indices_out = buffers.normal_indices;
        *d_leaf_texcoord_indices_out = buffers.texcoord_indices;

        // TODO: SBT indices handling
        *d_leaf_sbt_indices_out = nullptr;

        // Free temporary buffers
        cudaFree(d_buffers);
        cudaFree(buffers.vertex_count);
        cudaFree(buffers.face_count);
    }
} // namespace prayground

