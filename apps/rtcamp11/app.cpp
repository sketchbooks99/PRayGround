#include "app.h"
#include "textures.cuh"
#include "postprocess.cuh"
#if SVGF
#include "svgf.cuh"
#endif
#include <queue>

// ------------------------------------------------------------------
// Halton sequence for temporal jitter
// ------------------------------------------------------------------
static float halton(int index, int base) {
    float result = 0.0f;
    float f = 1.0f;
    int i = index;
    while (i > 0) {
        f /= base;
        result += f * (i % base);
        i = i / base;
    }
    return result;
}

static Vec2f getJitter(int frame_index) {
    // 8-tap Halton sequence for TAA
    const int pattern_size = 8;
    int idx = (frame_index % pattern_size) + 1;  // 1-based for Halton
    
    float jx = halton(idx, 2) - 0.5f;  // Range: [-0.5, 0.5]
    float jy = halton(idx, 3) - 0.5f;
    
    return Vec2f(jx, jy);
}

// ------------------------------------------------------------------
void App::initResultBufferOnDevice()
{
    m_params.frame = 0;
    m_bitmap.allocateDevicePtr();
    m_accum_buffer.allocateDevicePtr();
    m_float_bitmap.allocateDevicePtr();
    m_bloom_bitmap.allocateDevicePtr();

    m_params.result_buffer = (Vec4u*)m_bitmap.deviceData();
    m_params.accum_buffer = (Vec4f*)m_accum_buffer.deviceData();
    m_params.float_result_buffer = (Vec4f*)m_float_bitmap.deviceData();
    
    // Reset adaptive sampling buffers
    if (m_params.use_adaptive_sampling) {
        const size_t buffer_size = m_params.width * m_params.height * sizeof(Vec4f);
        CUDA_CHECK(cudaMemset(m_params.sum_buffer, 0, buffer_size));
        CUDA_CHECK(cudaMemset(m_params.sum_squared_buffer, 0, buffer_size));
        CUDA_CHECK(cudaMemset(m_params.sample_count_buffer, 0, m_params.width * m_params.height * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(m_params.converged_buffer, 0, m_params.width * m_params.height * sizeof(uint8_t)));
        // Note: converged_buffer removed (using probabilistic sampling instead)
    }

#if !SUBMISSION
    m_normal_bitmap.allocateDevicePtr();
    m_albedo_bitmap.allocateDevicePtr();
    m_uv_bitmap.allocateDevicePtr();

    m_params.normal_buffer = (Vec4f*)m_normal_bitmap.deviceData();
    m_params.albedo_buffer = (Vec4f*)m_albedo_bitmap.deviceData();
    m_params.uv_buffer = (Vec4f*)m_uv_bitmap.deviceData();
#endif

#if DENOISE || USE_SVGF
#endif

#if USE_SVGF
    m_svgf_output.allocateDevicePtr();
    m_position_bitmap.allocateDevicePtr();
    m_motion_bitmap.allocateDevicePtr();
    m_prev_position_bitmap.allocateDevicePtr();
    
    m_params.position_buffer = (Vec4f*)m_position_bitmap.deviceData();
    m_params.motion_buffer = (Vec4f*)m_motion_bitmap.deviceData();
    m_params.prev_position_buffer = (Vec4f*)m_prev_position_bitmap.deviceData();
#endif
}

// ------------------------------------------------------------------
void App::handleCameraUpdate()
{
    if (!is_camera_updated) return;

    is_camera_updated = false;

    m_scene.updateSBT(+(SBTRecordType::Raygen));

    initResultBufferOnDevice();
}

void App::resetMovie()
{
    is_camera_updated = true;
    m_frame_time = 0.0f;
}

template< typename tPair >
struct second_t {
    typename tPair::second_type operator()(const tPair& p) const { return p.second; }
};

template< typename tMap >
second_t< typename tMap::value_type > second(const tMap& m) { return second_t< typename tMap::value_type >(); }

void App::copyAreaEmitterToDevice()
{
    vector<AreaEmitterInfo> light_infos;
    transform(m_light_infos.begin(), m_light_infos.end(), back_inserter(light_infos), second(m_light_infos));

    d_light_infos.copyToDevice(light_infos);
    m_params.lights = d_light_infos.deviceData();
    m_params.n_lights = static_cast<uint32_t>(m_light_infos.size());
}

// ------------------------------------------------------------------
shared_ptr<TriangleMesh> App::buildVoronoiRockMesh(const VoronoiRockParams& params)
{
    // Call CUDA function to generate rock mesh
    Vec3f* d_vertices = nullptr;
    Vec3f* d_normals = nullptr;
    Vec2f* d_texcoords = nullptr;
    Vec3i* d_face_indices = nullptr;
    Vec3i* d_normal_indices = nullptr;
    Vec3i* d_texcoord_indices = nullptr;
    int vertex_count = 0;
    int face_count = 0;

    buildVoronoiRockCUDA(
        params,
        &d_vertices,
        &d_normals,
        &d_texcoords,
        &d_face_indices,
        &d_normal_indices,
        &d_texcoord_indices,
        &vertex_count,
        &face_count
    );

    // Transfer results from device to host
    vector<Vec3f> h_vertices(vertex_count);
    vector<Vec3f> h_normals(vertex_count);
    vector<Vec2f> h_texcoords(vertex_count);
    vector<Vec3i> h_face_indices(face_count);
    vector<Vec3i> h_normal_indices(face_count);
    vector<Vec3i> h_texcoord_indices(face_count);

    cudaMemcpy(h_vertices.data(), d_vertices, vertex_count * sizeof(Vec3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_normals.data(), d_normals, vertex_count * sizeof(Vec3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_texcoords.data(), d_texcoords, vertex_count * sizeof(Vec2f), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_face_indices.data(), d_face_indices, face_count * sizeof(Vec3i), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_normal_indices.data(), d_normal_indices, face_count * sizeof(Vec3i), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_texcoord_indices.data(), d_texcoord_indices, face_count * sizeof(Vec3i), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_vertices);
    cudaFree(d_normals);
    cudaFree(d_texcoords);
    cudaFree(d_face_indices);
    cudaFree(d_normal_indices);
    cudaFree(d_texcoord_indices);

    // Create Face structures
    vector<Face> faces(face_count);
    for (int i = 0; i < face_count; i++) {
        faces[i].vertex_id = h_face_indices[i];
        faces[i].normal_id = h_normal_indices[i];
        faces[i].texcoord_id = h_texcoord_indices[i];
    }
    
    // Calculate rock height (bottom is at Y=0 after CUDA normalization)
    float max_y = 0.0f;
    for (const auto& v : h_vertices) {
        if (v.y() > max_y) max_y = v.y();
    }
    float rock_height = max_y;
    
    // Shift rock so center is at Y=0 (instead of bottom at Y=0)
    // This makes placement easier: just place at terrain height
    float y_offset = -rock_height * 0.5f;
    for (auto& v : h_vertices) {
        v.y() += y_offset;
    }

    // Create TriangleMesh with centered vertices
    auto mesh = make_shared<TriangleMesh>();
    mesh->addVertices(h_vertices);
    mesh->addNormals(h_normals);
    mesh->addTexcoords(h_texcoords);
    mesh->addFaces(faces);

    mesh->calculateNormalFlat();

    return mesh;
}

// ------------------------------------------------------------------
App::TerrainMeshResult App::buildTerrainMesh(TerrainParams params)
{
    // Generate terrain using CUDA
    TerrainData terrain_data;
    buildTerrainMeshCUDA(params, terrain_data);
    
    // Convert to Face structures for TriangleMesh
    vector<Face> faces(terrain_data.faces.size());
    for (size_t i = 0; i < terrain_data.faces.size(); i++) {
        uint3 face = terrain_data.faces[i];
        faces[i].vertex_id = Vec3i(face.x, face.y, face.z);
        faces[i].normal_id = Vec3i(face.x, face.y, face.z);
        faces[i].texcoord_id = Vec3i(face.x, face.y, face.z);
    }
    
    // Create TriangleMesh
    auto mesh = make_shared<TriangleMesh>();
    mesh->addVertices(terrain_data.vertices);
    mesh->addNormals(terrain_data.normals);
    mesh->addTexcoords(terrain_data.texcoords);
    mesh->addFaces(faces);
    
    // Create heightmap texture (FloatBitmap)
    auto heightmap_bitmap = make_shared<FloatBitmap>();
    heightmap_bitmap->allocate(PixelFormat::GRAY, terrain_data.grid_width, terrain_data.grid_height);
    
    // Normalize heightmap to [0, 1] for texture
    float height_range = terrain_data.max_height - terrain_data.min_height;
    if (height_range < 1e-6f) height_range = 1.0f;  // Avoid division by zero
    
    vector<float> normalized_heightmap(terrain_data.heightmap.size());
    for (size_t i = 0; i < terrain_data.heightmap.size(); i++) {
        normalized_heightmap[i] = (terrain_data.heightmap[i] - terrain_data.min_height) / height_range;
    }
    
    // Set normalized heightmap data
    heightmap_bitmap->setData(normalized_heightmap.data(), 0, 0, terrain_data.grid_width, terrain_data.grid_width);
    
    return TerrainMeshResult{
        .mesh = mesh,
        .heightmap = heightmap_bitmap,
        .min_height = terrain_data.min_height,
        .max_height = terrain_data.max_height
    };
}


// ------------------------------------------------------------------
void convertTreeToCudaFormat(
    shared_ptr<TreeBranch> root,
    std::vector<CudaTreeBranch>& cuda_branches,
    std::vector<CudaTreeSegment>& cuda_segments
) {
    if (!root) return;

    cuda_branches.clear();
    cuda_segments.clear();

    // BFS（幅優先探索）でbranchを走査してフラット化
    std::queue<std::pair<shared_ptr<TreeBranch>, int>> queue;  // (branch, parent_idx)
    queue.push({ root, -1 });  // rootの親は-1

    while (!queue.empty()) {
        auto [branch, parent_idx] = queue.front();
        queue.pop();

        int current_branch_idx = cuda_branches.size();

        // CudaTreeBranchを作成
        CudaTreeBranch cuda_branch;
        cuda_branch.segment_start = cuda_segments.size();
        cuda_branch.segment_count = branch->segments.size();
        cuda_branch.child_start = -1;  // 後で設定
        cuda_branch.child_count = branch->children.size();
        cuda_branch.parent_idx = parent_idx;
        cuda_branch.generation = branch->generation;
        cuda_branch.length = branch->length;
        cuda_branch.radius = branch->radius;
        cuda_branch.is_terminal = branch->children.empty();

        cuda_branches.push_back(cuda_branch);

        // Segmentsを追加
        for (const auto& seg : branch->segments) {
            CudaTreeSegment cuda_seg;
            cuda_seg.position = seg.position;
            cuda_seg.direction = seg.direction;
            cuda_seg.normal = seg.normal;
            cuda_seg.binormal = seg.binormal;
            cuda_seg.radius = seg.radius;
            cuda_seg.generation = seg.generation;
            cuda_segments.push_back(cuda_seg);
        }

        // 子branchをqueueに追加
        for (const auto& child : branch->children) {
            queue.push({ child, current_branch_idx });
        }
    }
}

// ------------------------------------------------------------------
// Recursive helper: Build mesh for a single branch and all its children
// Returns a map of t_global -> ring_start for connecting children at correct positions
int App::buildBranchMeshRecursive(
    const Stem& stem,
    vector<Vec3f>& vertices,
    vector<Vec3f>& normals,
    vector<Vec2f>& texcoords,
    vector<Face>& faces,
    int radial_segments,
    int parent_ring_start,
    float parent_t_offset,
    const Vec3f* parent_right,
    const Vec3f* parent_tangent) 
{
    const auto& spline = stem.curve;
    if (!spline || spline->bezier_points.empty()) {
        return parent_ring_start;  // No geometry to add
    }
    
    int branch_start = vertices.size();
    int num_points = spline->bezier_points.size();
    
    int first_ring_start = -1;  // Track first ring of this branch
    int last_ring_start = -1;   // Track last ring of this branch
    
    // Store ring positions for child attachment (map: t_global -> ring_vertex_index)
    vector<pair<float, int>> ring_positions;
    
    // Store frame information for each ring (for passing to children)
    vector<pair<Vec3f, Vec3f>> ring_frames;  // {right, tangent} for each ring
    
    // Frame vectors for smooth tube (initialized on first ring, then propagated)
    Vec3f prev_right, prev_forward;
    
    // Inherit parent's frame if provided (for smooth connection)
    if (parent_right && parent_tangent) {
        prev_right = *parent_right;
        prev_forward = *parent_tangent;
    }
    
    // Build tube mesh for this branch - one ring per bezier point
    for (int pt = 0; pt < num_points; pt++) {
        const auto& bezier_pt = spline->bezier_points[pt];
        
        // Calculate t parameter along the branch [0, 1]
        float local_t = float(pt) / float(num_points - 1);
        float t_global = parent_t_offset + local_t;
        
        // Evaluate position and tangent at this point
        Vec3f pos = bezier_pt.co;
        Vec3f tangent = spline->evaluateTangent(local_t);
        
        // Ensure tangent is valid (not zero)
        if (length(tangent) < 1e-6f) {
            continue;  // Skip degenerate points
        }
        tangent = normalize(tangent);
        
        // Get radius at this point
        float radius = spline->evaluateRadius(local_t);
        
        // Create perpendicular frame
        Vec3f right, forward;
        if (pt == 0) {
            if (parent_right && parent_tangent) {
                // First point: inherit from parent
                right = *parent_right;
            } else {
                // First point: create initial frame
                Vec3f up = fabs(tangent.y()) < 0.99f ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0);
                right = normalize(cross(tangent, up));
            }
        } else {
            // Subsequent points: project previous right onto perpendicular plane
            right = prev_right - tangent * dot(prev_right, tangent);
            float len = length(right);
            if (len > 1e-6f) {
                right = right / len;
            } else {
                // Fallback
                Vec3f up = fabs(tangent.y()) < 0.99f ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0);
                right = normalize(cross(tangent, up));
            }
        }
        
        forward = normalize(cross(right, tangent));
        
        // Update for next iteration
        prev_right = right;
        prev_forward = tangent;
            
        // Update for next iteration
        prev_right = right;
        prev_forward = tangent;
        
        int ring_start = (int)vertices.size();
        
        // Create ring of vertices
        // Need radial_segments + 1 vertices for proper UV wrapping (0 and 1 at seam)
        for (int i = 0; i <= radial_segments; i++) {
            float angle = 2.0f * math::pi * (i % radial_segments) / radial_segments;
            float x = cos(angle);
            float z = sin(angle);
            
            Vec3f offset = (right * x + forward * z) * radius;
            Vec3f vertex_pos = pos + offset;
            Vec3f normal = normalize(offset);
            
            // U coordinate: 0 to 1 including both endpoints for seam
            float u_coord = float(i) / float(radial_segments);
            
            // Triangle wave for V coordinate: 0→1→0→1→0...
            // Makes texture repeat without seams (循環させる)
            float cycles = 2.0f;  // Number of cycles along the branch
            float v_raw = t_global * cycles;
            float v_fract = v_raw - floorf(v_raw);  // Get fractional part [0, 1)
            int cycle = (int)floorf(v_raw);
            
            // Triangle wave: even cycles go up (0→1), odd cycles go down (1→0)
            float v_coord = (cycle % 2 == 0) ? v_fract : (1.0f - v_fract);
            
            vertices.push_back(vertex_pos);
            normals.push_back(normal);
            texcoords.push_back(Vec2f(u_coord, v_coord));
        }
        
        // Remember first ring for parent connection
        if (first_ring_start == -1) {
            first_ring_start = ring_start;
        }
        last_ring_start = ring_start;
        
        // Store this ring position for child attachment
        ring_positions.push_back({t_global, ring_start});
        
        // Store frame at this ring for children to inherit
        ring_frames.push_back({right, tangent});
        
        // Connect to previous ring
        int prev_ring = (ring_start == first_ring_start && parent_ring_start >= 0) 
                        ? parent_ring_start  // Connect to parent's ring
                        : ring_start - (radial_segments + 1);  // Previous ring in this branch (+1 for seam vertex)
        
        if (ring_start > branch_start || parent_ring_start >= 0) {
            for (int i = 0; i < radial_segments; i++) {
                int next_i = i + 1;  // No modulo needed - we have the extra vertex
                
                int v0 = prev_ring + i;
                int v1 = prev_ring + next_i;
                int v2 = ring_start + next_i;
                int v3 = ring_start + i;
                
                // Two triangles per quad
                faces.push_back(Face{
                    Vec3i(v0, v1, v2),
                    Vec3i(v0, v1, v2),
                    Vec3i(v0, v1, v2)
                });
                faces.push_back(Face{
                    Vec3i(v0, v2, v3),
                    Vec3i(v0, v2, v3),
                    Vec3i(v0, v2, v3)
                });
            }
        }
    }
    
    // Recursively build all child branches at their correct attachment points
    for (const auto& child : stem.children) {
        // Find the ring closest to child's offset position
        // child->offset is in absolute length units, need to convert to t_global ratio
        float child_t = stem.length > 0 ? child->offset / stem.length : 0.0f;
        child_t = fminf(fmaxf(child_t, 0.0f), 1.0f);  // Clamp to [0, 1]
        
        // Find closest ring to this t value
        int attachment_ring = first_ring_start;
        float child_parent_t_offset = parent_t_offset;  // Default to current offset
        int ring_index = 0;
        float min_dist = FLT_MAX;
        for (int idx = 0; idx < ring_positions.size(); idx++) {
            const auto& [t, ring] = ring_positions[idx];
            float dist = fabs(t - child_t);
            if (dist < min_dist) {
                min_dist = dist;
                attachment_ring = ring;
                child_parent_t_offset = t;  // Use this ring's t_global as child's offset
                ring_index = idx;
            }
        }
        
        // Get frame at attachment point to pass to child
        const Vec3f& attach_right = ring_frames[ring_index].first;
        const Vec3f& attach_tangent = ring_frames[ring_index].second;
        
        // Build child branch independently (no parent ring connection)
        // Each child starts from its own Bezier curve position
        buildBranchMeshRecursive(*child, vertices, normals, texcoords, faces, 
                                  radial_segments, -1, child_parent_t_offset,
                                  &attach_right, &attach_tangent);
    }
    
    return last_ring_start;  // Return last ring for potential further connections
}

pair<shared_ptr<TriangleMesh>, shared_ptr<TriangleMesh>> App::buildTreeMesh(
    uint32_t& seed, Tree tree, TreeParam params, const vector<int>& leaf_texture_ids) {    
    // Generate tree structure (CPU side)
    tree.createBranches();
    
    const auto& stems = tree.getStems();
    const auto& leaves = tree.getLeaves();
    
    // DEBUG: Check tree structure recursively
    std::function<void(const Stem&, int, const string&)> printStemTree = 
        [&](const Stem& stem, int depth_level, const string& prefix) {
        
        for (int i = 0; i < stem.children.size() && i < 3; i++) {  // Limit to first 3 for brevity
            const auto& child = stem.children[i];
            printStemTree(*child, depth_level + 1, prefix + "  ");
        }
    };
    
    for (int i = 0; i < stems.size(); i++) {
        const auto& stem = stems[i];
        printStemTree(stem, 0, "  ");
    }
    
    // Convert Bezier curves to mesh using recursive approach
    auto tree_mesh = make_shared<TriangleMesh>();
    shared_ptr<TriangleMesh> leaf_mesh = nullptr;
    
    vector<Vec3f> vertices;
    vector<Vec3f> normals;
    vector<Vec2f> texcoords;
    vector<Face> faces;
    
    int radial_segments = 8;  // Number of vertices around the branch
    
    // Build mesh recursively starting from root stems
    // Each root stem and its entire subtree will be built in one traversal
    for (const auto& root_stem : stems) {
        buildBranchMeshRecursive(root_stem, vertices, normals, texcoords, faces, 
                                  radial_segments, -1, 0.0f, nullptr, nullptr);  // No parent frame
    }
    
    // Generate leaf mesh if leaves exist
    if (!leaves.empty()) {
        leaf_mesh = make_shared<TriangleMesh>();
        
        vector<Vec3f> leaf_vertices;
        vector<Vec3f> leaf_normals;
        vector<Vec2f> leaf_texcoords;
        vector<Face> leaf_faces;
        
        float leaf_size = params.leaf_scale * params.g_scale;
        float leaf_scale_x = params.leaf_scale_x;
        
        for (const auto& leaf : leaves) {
            int base_idx = (int)leaf_vertices.size();

            // Use Leaf class API: position(), direction(), right(), radius()
            Vec3f stem_center = leaf.position();
            Vec3f normal = leaf.direction();
            float stem_radius = leaf.radius();

            // Prefer using stored right vector if available; otherwise fallback to cross-product
            Vec3f right_dir = leaf.right();
            if (length(right_dir) < 1e-6f) {
                if (fabs(normal.y()) < 0.99f) {
                    right_dir = normalize(cross(normal, Vec3f(0, 1, 0)));
                } else {
                    right_dir = normalize(cross(normal, Vec3f(1, 0, 0)));
                }
            } else {
                right_dir = normalize(right_dir);
            }

            // Offset leaf base from stem center to stem surface
            // Use the right direction (perpendicular to leaf normal) to place leaf on stem surface
            Vec3f surface_offset = right_dir * stem_radius;
            Vec3f leaf_base = stem_center + surface_offset;

            // Leaf size is controlled by params; per-leaf scale removed (was in LeafData)
            Vec3f up = normal * leaf_size;
            Vec3f right = right_dir * leaf_size * leaf_scale_x;

            // Four corners of leaf quad - base at stem surface, extends upward
            Vec3f v0 = leaf_base - right;      // Bottom left
            Vec3f v1 = leaf_base + right;      // Bottom right
            Vec3f v2 = leaf_base + right + up; // Top right
            Vec3f v3 = leaf_base - right + up; // Top left
            leaf_vertices.push_back(v0);
            leaf_vertices.push_back(v1);
            leaf_vertices.push_back(v2);
            leaf_vertices.push_back(v3);


            for (int i = 0; i < 4; i++) {
                leaf_normals.push_back(normal);
            }

            leaf_texcoords.push_back(Vec2f(1, 1));
            leaf_texcoords.push_back(Vec2f(1, 0));
            leaf_texcoords.push_back(Vec2f(0, 0));
            leaf_texcoords.push_back(Vec2f(0, 1));

            // Two triangles for quad
            leaf_faces.push_back(Face{
                Vec3i(base_idx, base_idx + 1, base_idx + 2),
                Vec3i(base_idx, base_idx + 1, base_idx + 2),
                Vec3i(base_idx, base_idx + 1, base_idx + 2)
            });
            leaf_faces.push_back(Face{
                Vec3i(base_idx, base_idx + 2, base_idx + 3),
                Vec3i(base_idx, base_idx + 2, base_idx + 3),
                Vec3i(base_idx, base_idx + 2, base_idx + 3)
            });
        }
        
        if (!leaf_vertices.empty()) {
            leaf_mesh->addVertices(leaf_vertices);
            leaf_mesh->addNormals(leaf_normals);
            leaf_mesh->addTexcoords(leaf_texcoords);
            leaf_mesh->addFaces(leaf_faces);

            vector<uint32_t> sbt_indices;
            for (int i = 0; i < leaf_mesh->faces().size(); i++) {
                // Randomly select from provided leaf texture IDs
                int idx = rndInt(seed, 0, leaf_texture_ids.size() - 1);
                int tex_id = leaf_texture_ids[idx];
                sbt_indices.push_back(tex_id);
            }
            leaf_mesh->setSbtIndices(sbt_indices);
        }
    }
    
    if (!vertices.empty()) {
        tree_mesh->addVertices(vertices);
        tree_mesh->addNormals(normals);
        tree_mesh->addTexcoords(texcoords);
        tree_mesh->addFaces(faces);
    }

    return {tree_mesh, leaf_mesh};
}

// ------------------------------------------------------------------
void App::setup()
{
    using namespace std::chrono;

#if SUBMISSION
    // Start watchdog thread for time limit enforcement
    std::atomic<bool> rendering_complete(false);
    std::atomic<bool> time_limit_exceeded(false);
    std::thread watchdog_thread([&]() {
        auto watchdog_start = system_clock::now();
        while (!rendering_complete.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            auto elapsed = duration_cast<milliseconds>(system_clock::now() - watchdog_start).count() / 1000.0;

            // Display elapsed time (overwrite same line)
            std::cout << "\rElapsed: " << std::fixed << std::setprecision(1)
                << elapsed << "s / " << TIME_LIMIT << "s" << std::flush;

            if (elapsed >= TIME_LIMIT) {
                time_limit_exceeded.store(true);  // Signal time limit exceeded
                std::cout << "\n";
                pgLog(format("TIME_LIMIT ({:.2f}s) reached! Force exiting...", TIME_LIMIT));
                pgExit();
                std::exit(0);  // Force exit immediately at 180s
            }
        }
        });
    watchdog_thread.detach();  // Detach so it runs independently

    system_clock::time_point start_time = system_clock::now();
#endif

    n_frame = static_cast<int>(VIDEO_LENGTH * FPS);
    m_interval = 1.0f / FPS;
    m_frame_time = 0.0f;
    m_camera_ease = EaseType::InOutSine;

    m_stream = 0;
    CUDA_CHECK(cudaFree(0));

    // Initialize OptiX context
    OPTIX_CHECK(optixInit());
    m_ctx.disableValidation();
    m_ctx.create();

    // Initialize pipeline
    m_ppl.setLaunchVariableName("params");
    m_ppl.setNumPayloads(5);
    m_ppl.setNumAttributes(5);
    m_ppl.setContinuationCallableDepth(5);
    m_ppl.setDirectCallableDepth(5);
    m_ppl.setMaxTraversableGraphDepth(3);

    // Create module
    Module module = m_ppl.createModuleFromOptixIr(m_ctx, "rtcamp11_generated_kernels.cu.optixir");

    // Initialize buffers
    const int width = pgGetWidth();
    const int height = pgGetHeight();
    m_bitmap.allocate(PixelFormat::RGBA, width, height);
    m_accum_buffer.allocate(PixelFormat::RGBA, width, height);
    m_float_bitmap.allocate(PixelFormat::RGBA, width, height);
    m_bloom_bitmap.allocate(PixelFormat::RGBA, width, height);

    // Allocate bloom effect buffers (Vec4f version)
    const size_t buffer_size = width * height * sizeof(Vec4f);
    CUDA_CHECK(cudaMalloc(&d_bloom_temp1, buffer_size));
    CUDA_CHECK(cudaMalloc(&d_bloom_temp2, buffer_size));
    CUDA_CHECK(cudaMalloc(&d_firefly_temp, buffer_size));

    // Allocate adaptive sampling buffers
    CUDA_CHECK(cudaMalloc(&m_params.sum_buffer, buffer_size));
    CUDA_CHECK(cudaMalloc(&m_params.sum_squared_buffer, buffer_size));
    CUDA_CHECK(cudaMalloc(&m_params.sample_count_buffer, width * height * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&m_params.converged_buffer, width * height * sizeof(uint8_t)));
    
    // Initialize adaptive sampling buffers to zero
    CUDA_CHECK(cudaMemset(m_params.sum_buffer, 0, buffer_size));
    CUDA_CHECK(cudaMemset(m_params.sum_squared_buffer, 0, buffer_size));
    CUDA_CHECK(cudaMemset(m_params.sample_count_buffer, 0, width * height * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(m_params.converged_buffer, 0, width * height * sizeof(uint8_t)));

#if !SUBMISSION
    m_normal_bitmap.allocate(PixelFormat::RGBA, width, height);
    m_albedo_bitmap.allocate(PixelFormat::RGBA, width, height);
    m_uv_bitmap.allocate(PixelFormat::RGBA, width, height);
#endif

    initResultBufferOnDevice();

    // Configuration of launch parameters
    m_params.width = width;
    m_params.height = height;
    m_params.samples_per_launch = 1;
    m_params.frame = 0u;
    m_params.max_depth = 10u;
    m_params.enable_mis = enable_mis;
    
    // Stratified sampling: Calculate grid dimension based on total SPP
    // SPP = 16 → 4x4, SPP = 64 → 8x8, SPP = 256 → 16x16, etc.
    m_params.stratified_dim = static_cast<uint32_t>(sqrtf(static_cast<float>(SPP)));
    
    // Adaptive sampling configuration
    m_params.use_adaptive_sampling = ADAPTIVE_SAMPLING;  // Enable/disable adaptive sampling
    m_params.adaptive_min_samples = ADAPTIVE_MIN_SAMPLES;  // Minimum samples before convergence check
    
    // MIS configuration
    m_params.mis_heuristic = MISHeuristic::PowerBeta2;  // Balance or PowerBeta2
    m_params.use_multi_light_sampling = true;  // Sample both area+env lights
    
    m_params.white = 0.5f;

    // Bloom parameters
    bloom_threshold = 0.1f;
    bloom_intensity = 0.7f;
    bloom_radius = 10.0f;
    bloom_sigma = 10.0f;

    // Setup scene
    AppScene::AccelSettings accel_settings = {
        .allow_accel_compaction = true,
        .allow_accel_update = true
    };
    m_scene.setup(accel_settings);

    Vec3f final_point = Vec3f(161.6f, 18.17f, -80.72f);
    Vec3f final_look = Vec3f(182.31f, 15.81f, -93.42f);
    Vec3f move_dir = normalize(final_point - Vec3f(0.0f, 18.17f, 0.0f));
    Vec3f look_dir = normalize(final_look - Vec3f(0.0f, 15.81f, 0.0f));
    // Vec3f first_point = Vec3f(0.0f, 18.17f, 0.0f) - move_dir * 300.0f;
    // Vec3f first_look = Vec3f(0.0f, 15.81f, 0.0f) - look_dir * 200.0f;
    Vec3f first_point = Vec3f(-306.5f, 18.2f, 116.5f);
    Vec3f first_look = Vec3f(-213.1f, 15.8f, 80.6f);
    pgLog("First point:", first_point);
    pgLog("First look:", first_look);
    // Camera positions
    m_cam_points = {
        {first_point, 0.0f},
        {final_point, VIDEO_LENGTH * 0.9f}
    };
    m_look_points = {
        {first_look, 0.0f},
        {final_look, VIDEO_LENGTH * 0.9f}
    };

    // Camera settings
    shared_ptr<Camera> camera = make_shared<Camera>();
    camera->setOrigin(final_point);
    camera->setLookat(final_look);
    camera->setUp(0, 1, 0);
    camera->setFov(40);
    camera->setAspect((float)width / height);
#if !SUBMISSION
    camera->enableTracking(pgGetCurrentWindow());
#endif
    m_scene.setCamera(camera);

    // Ray generation program
    ProgramGroup raygen_prg = m_ppl.createRaygenProgram(m_ctx, module, "__raygen__pinhole");
    m_scene.bindRaygenProgram(raygen_prg);

    // Callable programs 
    struct Callable {
        Callable(const pair<ProgramGroup, uint32_t>& callable)
            : prg(callable.first), ID(callable.second) {};
        ProgramGroup prg;
        uint32_t ID;
    };

    auto setupCallable = [&](const string& dc_name, const string& cc_name) {
        Callable callable = m_ppl.createCallablesProgram(m_ctx, module, dc_name, cc_name);
        m_scene.bindCallablesProgram(callable.prg);
        return callable.ID;
    };
    
    // Texture callables
    auto bitmap_id = setupCallable("__direct_callable__bitmap", "");
    auto checker_id = setupCallable("__direct_callable__checker", "");
    auto constant_id = setupCallable("__direct_callable__constant", "");
    auto procedural_wooden_id = setupCallable("__direct_callable__procedural_wooden", "");
    auto terrain_id = setupCallable("__direct_callable__terrain_heightmap", "");
    auto uv_id = setupCallable("__direct_callable__uv", "");
    
    // Register environment map importance sampling callable functions
    auto envmap_sample_id = setupCallable("__direct_callable__sample_envmap", "");
    auto envmap_pdf_id = setupCallable("__direct_callable__pdf_envmap", "");
    
    // Bake procedural texture and build CDF for importance sampling
    StarNightTexture::Data star_night_data{
        .base_color = Vec4f(0.0f, 0.0f, 0.0001f, 1.0f),
        .noise_data = RandomNoise::Data{
            .seed = 1234,
            .width = 4096, 
            .height = 4096, 
            .depth = 1
        },
        .star_threshold = 0.9996f,
        .star_intensity = 5.0f,
        .moon_dir = normalize(Vec3f(1.0f, 0.2f, -1.0f)),
        .moon_intensity = 100.0f
    };
    
    // Step 1: Bake to device buffer using CUDA kernel (Vec4f for RGBA)
    uint32_t envmap_width = 4096;
    uint32_t envmap_height = 4096;
    Vec4f* d_envmap_colors = bakeProceduralTextureToDevice(
        star_night_data,
        envmap_width,
        envmap_height
    );
    
    // Step 2: Create FloatBitmap from device buffer for texture sampling
    auto envmap_bitmap = make_shared<FloatBitmap>();
    envmap_bitmap->allocate(PixelFormat::RGBA, envmap_width, envmap_height);
    envmap_bitmap->allocateDevicePtr();
    
    // Directly copy from d_envmap_colors to device bitmap (no host copy needed)
    CUDA_CHECK(cudaMemcpy(envmap_bitmap->deviceData(), d_envmap_colors, 
                         sizeof(Vec4f) * envmap_width * envmap_height, 
                         cudaMemcpyDeviceToDevice));

    envmap_bitmap->copyFromDevice();
    auto envmap_texture = make_shared<FloatBitmapTexture>(envmap_bitmap, bitmap_id);
    envmap_texture->copyToDevice();
    
    // Step 3: Build CDF from device buffer
    auto envmap_data = buildEnvmapSamplingDataFromDevice(
        d_envmap_colors,
        4096, 4096,
        bitmap_id,
        envmap_bitmap->deviceData()
    );
    envmap_data.texture_id = bitmap_id;
    envmap_data.texture_data = envmap_texture->devicePtr();

    CUDABuffer<EnvmapSamplingData> d_envmap_data;
    d_envmap_data.copyToDevice(&envmap_data, sizeof(EnvmapSamplingData));

    m_params.envmap_sampling_data = d_envmap_data.deviceData();
    m_params.envmap_sample_id = envmap_sample_id;
    m_params.envmap_pdf_id = envmap_pdf_id;
    m_params.envmap_texture_id = bitmap_id;
    m_params.envmap_texture_data = envmap_texture->devicePtr();
    
    // Free intermediate buffer (data is now in envmap_bitmap)
    CUDA_CHECK(cudaFree(d_envmap_colors));

    // Miss program & environemnt map
    array<ProgramGroup, NRay> miss_prgs;
    miss_prgs[0] = m_ppl.createMissProgram(m_ctx, module, "__miss__envmap");
    miss_prgs[1] = m_ppl.createMissProgram(m_ctx, module, "__miss__shadow");
    m_scene.bindMissPrograms(miss_prgs);
    m_scene.setEnvmap(envmap_texture);

    // Hitgroup programs
    array<ProgramGroup, NRay> mesh_prgs;
    mesh_prgs[0] = m_ppl.createHitgroupProgram(m_ctx, module, "__closesthit__mesh", "", "__anyhit__mesh_opacity");
    mesh_prgs[1] = m_ppl.createHitgroupProgram(m_ctx, module, "__closesthit__shadow", "", "__anyhit__mesh_opacity");
    // Sphere
    array<ProgramGroup, NRay> sphere_prgs;
    sphere_prgs[0] = m_ppl.createHitgroupProgram(m_ctx, module, "__closesthit__sphere", "__intersection__sphere");
    sphere_prgs[1] = m_ppl.createHitgroupProgram(m_ctx, module, "__closesthit__shadow", "__intersection__sphere");
    // Plane
    array<ProgramGroup, NRay> plane_prgs;
    plane_prgs[0] = m_ppl.createHitgroupProgram(m_ctx, module, "__closesthit__plane", "__intersection__plane");
    plane_prgs[1] = m_ppl.createHitgroupProgram(m_ctx, module, "__closesthit__shadow", "__intersection__plane");
    // Curves
    array<ProgramGroup, NRay> curve_prgs;
    Module curve_module = m_ppl.createBuiltinIntersectionModule(m_ctx, OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE);
    curve_prgs[0] = m_ppl.createHitgroupProgram(m_ctx, { module, "__closesthit__curves" }, { curve_module, "" });
    curve_prgs[1] = m_ppl.createHitgroupProgram(m_ctx, { module, "__closesthit__shadow" }, { curve_module, "" });

    // Create callables program for surfaces
    auto setupSurfaceCallable = [&](const string& dc_sample, const string& cc_bsdf, const string& dc_pdf) {
        uint32_t sample_bsdf_id = setupCallable(dc_sample, cc_bsdf);
        uint32_t pdf_id = setupCallable(dc_pdf, "");

        return SurfaceCallableID{ sample_bsdf_id, sample_bsdf_id, pdf_id };
    };

    auto diffuse_id = setupSurfaceCallable(
        "__direct_callable__sample_diffuse",
        "__continuation_callable__bsdf_diffuse",
        "__direct_callable__pdf_diffuse"
    );

    auto dielectric_id = setupSurfaceCallable(
        "__direct_callable__sample_dielectric",
        "__continuation_callable__bsdf_dielectric",
        "__direct_callable__pdf_dielectric"
    );

    auto disney_id = setupSurfaceCallable(
        "__direct_callable__sample_disney",
        "__continuation_callable__bsdf_disney",
        "__direct_callable__pdf_disney"
    );

    uint32_t area_id = setupCallable("__direct_callable__area_emitter", "");
    SurfaceCallableID area_emitter_id = {
        area_id,
        area_id,
        area_id
    };

    uint32_t seed = tea<4>(0, 0);

    // Light sampling callables for MIS
    uint32_t sphere_light_sample_id = setupCallable("__direct_callable__sample_sphere_light", "");
    uint32_t sphere_light_pdf_id = setupCallable("__direct_callable__pdf_sphere_light", "");

    // Textures
    auto procedural_wooden_texture = make_shared<ProceduralWoodenTexture>(
        Vec4f(0.76f, 0.60f, 0.42f, 1.0f), Vec4f(0.4f, 0.2f, 0.1f, 1.0f),
        0.1f, 20.0f, 0.2f,
        procedural_wooden_id
    );
    auto checker_texture = make_shared<CheckerTexture>(
        Vec4f(0.2f, 0.2f, 0.2f, 1.0f), Vec4f(0.8f, 0.8f, 0.8f, 1.0f), 10.0f, checker_id
    );
    auto grey_texture = make_shared<ConstantTexture>(Vec4f(0.5f, 0.5f, 0.5f, 1.0f), constant_id);
    auto white_texture = make_shared<ConstantTexture>(Vec4f(1.0f), constant_id);
    auto winered_checker_texture = make_shared<CheckerTexture>(
        Vec4f(0.5f, 0.0f, 0.0f, 1.0f), Vec4f(0.2f, 0.2f, 0.2f, 1.0f), 5.0f, checker_id
    );

    // Procedural rock textures
    RockTexture rock_texture;
    rock_texture.noise1 = {
        .depth = 7,
        .scale = 5.0f,
        .amplitude = 1.0f
    };
    rock_texture.noise2 = {
        .depth = 7,
        .scale = 10.0f,
        .amplitude = 0.5f
    };
    
    // Clear any previous CUDA errors before texture generation
    cudaGetLastError();
    
    float* d_rock_noise_data = bakeRockTexture(seed, rock_texture, 1024, 1024);
    
    if (d_rock_noise_data == nullptr) {
        return;
    }

    uint32_t rock_bump_width = 1024;
    uint32_t rock_bump_height = 1024;

    Vec4f* d_rock_bumpmap_data = createBumpTextureFromHeightmap(
        d_rock_noise_data,
        rock_bump_width, rock_bump_height,
        0.3f  // bump strength
    );

    if (d_rock_bumpmap_data == nullptr) {
        CUDA_CHECK(cudaFree(d_rock_noise_data));
        return;
    }

    Vec4f* h_rock_bumpmap_data = new Vec4f[rock_bump_width * rock_bump_height];
    CUDA_CHECK(cudaMemcpy(h_rock_bumpmap_data, d_rock_bumpmap_data, sizeof(Vec4f) * rock_bump_width * rock_bump_height, cudaMemcpyDeviceToHost));
    auto rock_bumpmap_bitmap = make_shared<FloatBitmap>();
    rock_bumpmap_bitmap->allocate(PixelFormat::RGBA, rock_bump_width, rock_bump_height);
    rock_bumpmap_bitmap->setData((float*)h_rock_bumpmap_data, 0, 0, rock_bump_width, rock_bump_height);
    auto rock_bumpmap_texture = make_shared<FloatBitmapTexture>(rock_bumpmap_bitmap, bitmap_id);
    rock_bumpmap_texture->copyToDevice();
    delete[] h_rock_bumpmap_data;
    CUDA_CHECK(cudaFree(d_rock_noise_data));
    CUDA_CHECK(cudaFree(d_rock_bumpmap_data));
    
    // ===== Tree Bark Textures =====
    // Generate 3 types of bark textures
    
    // 1. Rough Bark (Worley crackle - ごつごつ)
    TreeBarkTexture rough_bark;
    rough_bark.type = BarkType::ROUGH;
    rough_bark.rough.cell_scale = 1.0f;
    rough_bark.rough.vertical_stretch = 1.0f;
    rough_bark.rough.crack_depth = 0.3f;
    rough_bark.rough.crack_threshold = 0.15f;
    rough_bark.bump_strength = 1.0f;
    rough_bark.seed = tea<4>(seed, 1001);
    
    printf("[DEBUG APP] rough_bark.bump_strength = %.4f\n", rough_bark.bump_strength);
    
    uint32_t bark_width = 1024;
    uint32_t bark_height = 1024;
    
    float* d_rough_bark_heightmap = bakeTreeBarkTexture(rough_bark, bark_width, bark_height);
    
    // Debug: Check heightmap values
    float* h_rough_heightmap_debug = new float[100];
    CUDA_CHECK(cudaMemcpy(h_rough_heightmap_debug, d_rough_bark_heightmap, 
                         sizeof(float) * 100, cudaMemcpyDeviceToHost));
    printf("[DEBUG Rough Bark] First 10 heightmap values: ");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", h_rough_heightmap_debug[i]);
    }
    printf("\n");
    delete[] h_rough_heightmap_debug;

    Vec4f* d_rough_bark_bumpmap = createBumpTextureFromHeightmap(
        d_rough_bark_heightmap, bark_width, bark_height, -2.0f  // Negative for inverted normals (凹凸逆転)
    );
    
    Vec4f* h_rough_bark_data = new Vec4f[bark_width * bark_height];
    CUDA_CHECK(cudaMemcpy(h_rough_bark_data, d_rough_bark_bumpmap, 
                         sizeof(Vec4f) * bark_width * bark_height, cudaMemcpyDeviceToHost));
    auto rough_bark_bitmap = make_shared<FloatBitmap>();
    rough_bark_bitmap->allocate(PixelFormat::RGBA, bark_width, bark_height);
    rough_bark_bitmap->setData((float*)h_rough_bark_data, 0, 0, bark_width, bark_height);
    auto rough_bark_texture = make_shared<FloatBitmapTexture>(rough_bark_bitmap, bitmap_id);
    rough_bark_texture->copyToDevice();
    delete[] h_rough_bark_data;
    CUDA_CHECK(cudaFree(d_rough_bark_heightmap));
    CUDA_CHECK(cudaFree(d_rough_bark_bumpmap));
    
    // 2. Aged Bark (Layered FBM - 年季入り)
    TreeBarkTexture aged_bark;
    aged_bark.type = BarkType::AGED;
    aged_bark.aged.octaves = 8;
    aged_bark.aged.scale = 8.0f;
    aged_bark.aged.warp_strength = 0.3f;
    aged_bark.aged.vertical_bias = 0.5f;
    aged_bark.bump_strength = 1.0f;
    aged_bark.seed = tea<4>(seed, 1002);
    
    printf("[DEBUG APP] aged_bark.bump_strength = %.4f\n", aged_bark.bump_strength);
    
    float* d_aged_bark_heightmap = bakeTreeBarkTexture(aged_bark, bark_width, bark_height);
    
    // Debug: Check heightmap values
    float* h_aged_heightmap_debug = new float[100];
    CUDA_CHECK(cudaMemcpy(h_aged_heightmap_debug, d_aged_bark_heightmap, 
                         sizeof(float) * 100, cudaMemcpyDeviceToHost));
    delete[] h_aged_heightmap_debug;
    
    Vec4f* d_aged_bark_bumpmap = createBumpTextureFromHeightmap(
        d_aged_bark_heightmap, bark_width, bark_height, 1.5f  // Increased for stronger bumps
    );
    
    Vec4f* h_aged_bark_data = new Vec4f[bark_width * bark_height];
    CUDA_CHECK(cudaMemcpy(h_aged_bark_data, d_aged_bark_bumpmap, 
                         sizeof(Vec4f) * bark_width * bark_height, cudaMemcpyDeviceToHost));
    auto aged_bark_bitmap = make_shared<FloatBitmap>();
    aged_bark_bitmap->allocate(PixelFormat::RGBA, bark_width, bark_height);
    aged_bark_bitmap->setData((float*)h_aged_bark_data, 0, 0, bark_width, bark_height);
    auto aged_bark_texture = make_shared<FloatBitmapTexture>(aged_bark_bitmap, bitmap_id);
    aged_bark_texture->copyToDevice();
    delete[] h_aged_bark_data;
    CUDA_CHECK(cudaFree(d_aged_bark_heightmap));
    CUDA_CHECK(cudaFree(d_aged_bark_bumpmap));
    
    // 3. Smooth Bark (Color texture - 滑らか)
    TreeBarkTexture smooth_bark;
    smooth_bark.type = BarkType::SMOOTH;
    smooth_bark.smooth.flow_scale = 3.0f;
    smooth_bark.smooth.flow_strength = 0.3f;
    smooth_bark.smooth.ripple_frequency = 30.0f;
    smooth_bark.smooth.smoothness = 1.0f;
    smooth_bark.bump_strength = 1.0f;
    smooth_bark.seed = tea<4>(seed, 1003);
    
    // Generate color texture (not bumpmap) for smooth bark
    Vec4f* d_smooth_bark_color = bakeSmoothBarkColorTexture(smooth_bark, bark_width, bark_height);
    
    Vec4f* h_smooth_bark_data = new Vec4f[bark_width * bark_height];
    CUDA_CHECK(cudaMemcpy(h_smooth_bark_data, d_smooth_bark_color, 
                         sizeof(Vec4f) * bark_width * bark_height, cudaMemcpyDeviceToHost));
    auto smooth_bark_bitmap = make_shared<FloatBitmap>();
    smooth_bark_bitmap->allocate(PixelFormat::RGBA, bark_width, bark_height);
    smooth_bark_bitmap->setData((float*)h_smooth_bark_data, 0, 0, bark_width, bark_height);
    auto smooth_bark_texture = make_shared<FloatBitmapTexture>(smooth_bark_bitmap, bitmap_id);
    smooth_bark_texture->copyToDevice();
    delete[] h_smooth_bark_data;
    CUDA_CHECK(cudaFree(d_smooth_bark_color));

    auto leaf1 = make_shared<BitmapTexture>("foliage_10.png", bitmap_id);
    auto leaf2 = make_shared<BitmapTexture>("foliage_15.png", bitmap_id);
    auto leaf3 = make_shared<BitmapTexture>("foliage_24.png", bitmap_id);
    auto leaf4 = make_shared<BitmapTexture>("foliage_52.png", bitmap_id);
    auto leaf5 = make_shared<BitmapTexture>("foliage_82.png", bitmap_id);
    auto leafs = vector<shared_ptr<BitmapTexture>>{ leaf1, leaf2, leaf3, leaf4, leaf5 };

    auto uv_texture = make_shared<ConstantTexture>(Vec3f(0.0f), uv_id);

    // Surfaces
    auto floor_diffuse = make_shared<Diffuse>(diffuse_id, checker_texture);
    auto tree_diffuse = make_shared<Diffuse>(diffuse_id, procedural_wooden_texture, false);

    // Generate terrain first (before placing trees/rocks)
    TerrainParams terrain_params;
    terrain_params.terrain_size = 1000.0f;
    terrain_params.height_scale = 100.0f;
    auto terrain_result = buildTerrainMesh(terrain_params);
    auto terrain_mesh = terrain_result.mesh;
    auto terrain_heightmap = terrain_result.heightmap;
    auto terrain_texture = make_shared<FloatBitmapTexture>(terrain_heightmap, terrain_id);
    auto terrain_diffuse = make_shared<Diffuse>(diffuse_id, terrain_texture);
    
    // Terrain offset
    const Vec3f terrain_offset(0, -terrain_params.height_scale / 2.0f, 0);
    m_scene.addObject("terrain", terrain_mesh, terrain_diffuse, mesh_prgs, Matrix4f::translate(terrain_offset));
    
    // Use actual min/max height from terrain generation
    const float terrain_min_height = terrain_result.min_height;
    const float terrain_max_height = terrain_result.max_height;
    
    // Helper lambda to get terrain height at world position (x, z)
    auto getTerrainHeight = [&](float world_x, float world_z) -> float {
        // Convert world coordinates to terrain local coordinates
        float local_x = world_x - terrain_offset.x();
        float local_z = world_z - terrain_offset.z();
        
        // Convert to UV coordinates (terrain is centered at origin, size is terrain_size)
        float half_size = terrain_params.terrain_size * 0.5f;
        float u = (local_x + half_size) / terrain_params.terrain_size;
        float v = (local_z + half_size) / terrain_params.terrain_size;
        
        // Clamp to valid range
        u = clamp(u, 0.0f, 1.0f);
        v = clamp(v, 0.0f, 1.0f);
        
        // Sample heightmap (bilinear interpolation)
        int width = terrain_heightmap->width();
        int height = terrain_heightmap->height();
        float fx = u * (width - 1);
        float fy = v * (height - 1);
        int x0 = static_cast<int>(fx);
        int y0 = static_cast<int>(fy);
        int x1 = std::min(x0 + 1, width - 1);
        int y1 = std::min(y0 + 1, height - 1);
        
        float tx = fx - x0;
        float ty = fy - y0;
        
        // Get normalized height values [0,1] from bitmap
        auto getData = [&](int x, int y) -> float {
            return get<float>(terrain_heightmap->at(x, y));  // GRAY format, single channel
        };
        
        float h00 = getData(x0, y0);
        float h10 = getData(x1, y0);
        float h01 = getData(x0, y1);
        float h11 = getData(x1, y1);
        
        // Bilinear interpolation
        float h0 = h00 * (1.0f - tx) + h10 * tx;
        float h1 = h01 * (1.0f - tx) + h11 * tx;
        float normalized_height = h0 * (1.0f - ty) + h1 * ty;
        
        // Convert normalized height [0,1] back to actual height
        // normalized = (height - min) / (max - min)
        // height = normalized * (max - min) + min
        float height_range = terrain_max_height - terrain_min_height;
        float actual_height = normalized_height * height_range + terrain_min_height;
        
        return terrain_offset.y() + actual_height;
    };
    
    Vec3f camera_start(first_point);
    Vec3f camera_end(final_point);
    Vec3f camera_direction = camera_end - camera_start;
    float path_length = length(camera_direction);
    
    // Normalize direction for forward axis
    Vec3f forward = normalize(camera_direction);
    
    // Create perpendicular axis for grid (right vector)
    Vec3f up(0.0f, 1.0f, 0.0f);
    Vec3f right = normalize(cross(forward, up));
    
    const int grid_rows = 8;       // Along camera path
    const int grid_cols = 8;       // Perpendicular to path
    const float grid_spacing = 50.0f;  // Distance between grid points
    const float perturbation = 8.0f;   // Random offset ±8 units
    
    vector<Vec2f> tree_positions_2d;
    
    // Define exclusion zones (sacred areas where trees should not be placed)
    struct ExclusionZone {
        Vec2f center;
        float radius;
    };
    
    vector<ExclusionZone> exclusion_zones;
    
    // Sacred zone 1: Final camera position
    exclusion_zones.push_back({Vec2f(final_point.x(), final_point.z()), 150.0f});
    
    // Sacred zone 2: Bunny1 vicinity (in front of bunny)
    exclusion_zones.push_back({Vec2f(m_bunny1_pos.x(), m_bunny1_pos.z()), 200.0f});
    
    // Sacred zone 3: Bunny2 vicinity
    exclusion_zones.push_back({Vec2f(m_bunny2_pos.x(), m_bunny2_pos.z()), 200.0f});
    
    // Sacred zone 4: Bunny3 vicinity
    exclusion_zones.push_back({Vec2f(m_bunny3_pos.x(), m_bunny3_pos.z()), 200.0f});
    
    // Helper to check if position is in exclusion zone
    auto isInExclusionZone = [&](const Vec2f& pos) -> bool {
        for (const auto& zone : exclusion_zones) {
            Vec2f diff = pos - zone.center;
            float dist_sq = diff.x() * diff.x() + diff.y() * diff.y();
            if (dist_sq < zone.radius * zone.radius) {
                return true;
            }
        }
        return false;
    };

    m_bunny1_pos = Vec3f(270.0f, -7.0f, -210.0f);
    m_bunny2_pos = Vec3f(350.0f, -7.0f, -200.0f);
    m_bunny3_pos = Vec3f(280.0f, -9.0f, -110.0f);
    
    // Special tree: Place one tree diagonally behind bunny2 and bunny3
    {
        // Calculate position between bunny2 and bunny3, then offset backward
        Vec3f bunny_midpoint = (m_bunny2_pos + m_bunny3_pos) * 0.5f;
        
        // Move backward (opposite of forward direction) by ~80 units
        Vec3f special_tree_3d = bunny_midpoint + forward * 100.0f;
        
        Vec2f special_tree_pos(special_tree_3d.x(), special_tree_3d.z());
        tree_positions_2d.push_back(special_tree_pos);
        
        pgLog("Special tree placed at:", special_tree_3d);
    }
    
    // Generate grid aligned with camera path
    for (int row = 0; row < grid_rows; row++) {
        for (int col = 0; col < grid_cols; col++) {
            // Position along camera path (row) and perpendicular (col)
            float forward_dist = (row - grid_rows * 0.5f) * grid_spacing;
            float right_dist = (col - grid_cols * 0.5f) * grid_spacing;
            
            // Calculate 3D position
            Vec3f base_pos = forward * forward_dist + right * right_dist;
            
            // Add random perturbation
            float perturb_x = (rnd(seed) - 0.5f) * perturbation * 2.0f;
            float perturb_z = (rnd(seed) - 0.5f) * perturbation * 2.0f;
            
            float pos_x = base_pos.x() + perturb_x;
            float pos_z = base_pos.z() + perturb_z;
            
            tree_positions_2d.push_back(Vec2f(pos_x, pos_z));
        }
    }
    
    // Add random trees around the grid (19 trees, since we already have 1 special tree)
    const int num_random_trees = 19;
    const float grid_half_width = grid_cols * 0.5f * grid_spacing;
    const float grid_half_depth = grid_rows * 0.5f * grid_spacing;
    const float outer_min_distance = grid_spacing * 0.5f;  // Minimum distance from grid edge
    const float outer_max_distance = grid_spacing * 3.0f;  // Maximum distance from grid edge
    
    int attempts = 0;
    int placed_random_trees = 0;
    const int max_attempts = 1000;  // Prevent infinite loop
    
    while (placed_random_trees < num_random_trees && attempts < max_attempts) {
        seed = tea<4>(attempts + 10000, 42);  // Different seed offset for random trees
        
        // Choose random side: 0=front, 1=back, 2=left, 3=right
        int side = static_cast<int>(rnd(seed) * 4.0f);
        
        float pos_x, pos_z;
        
        if (side == 0) {
            // Front side (positive forward direction)
            float dist = grid_half_depth + outer_min_distance + rnd(seed) * (outer_max_distance - outer_min_distance);
            pos_x = (rnd(seed) - 0.5f) * grid_half_width * 3.0f;  // Wider spread
            Vec3f pos_3d = forward * dist + right * pos_x;
            pos_x = pos_3d.x();
            pos_z = pos_3d.z();
        }
        else if (side == 1) {
            // Back side (negative forward direction)
            float dist = -(grid_half_depth + outer_min_distance + rnd(seed) * (outer_max_distance - outer_min_distance));
            pos_x = (rnd(seed) - 0.5f) * grid_half_width * 3.0f;
            Vec3f pos_3d = forward * dist + right * pos_x;
            pos_x = pos_3d.x();
            pos_z = pos_3d.z();
        }
        else if (side == 2) {
            // Left side (negative right direction)
            float dist = -(grid_half_width + outer_min_distance + rnd(seed) * (outer_max_distance - outer_min_distance));
            float forward_offset = (rnd(seed) - 0.5f) * grid_half_depth * 3.0f;
            Vec3f pos_3d = right * dist + forward * forward_offset;
            pos_x = pos_3d.x();
            pos_z = pos_3d.z();
        }
        else {
            // Right side (positive right direction)
            float dist = grid_half_width + outer_min_distance + rnd(seed) * (outer_max_distance - outer_min_distance);
            float forward_offset = (rnd(seed) - 0.5f) * grid_half_depth * 3.0f;
            Vec3f pos_3d = right * dist + forward * forward_offset;
            pos_x = pos_3d.x();
            pos_z = pos_3d.z();
        }
        
        Vec2f candidate_pos(pos_x, pos_z);
        
        // Check if position is in exclusion zone
        if (!isInExclusionZone(candidate_pos)) {
            tree_positions_2d.push_back(candidate_pos);
            placed_random_trees++;
        }
        
        attempts++;
    }
    
    pgLog("Random trees placed:", placed_random_trees, "out of", num_random_trees, "attempts:", attempts);
    
    float tree_height_offset = -3.0f;

    // Place trees at selected positions
    for (size_t i = 0; i < tree_positions_2d.size(); i++) {
        seed = tea<4>(i, 1);

        // Create Tree object with parameters
        Tree tree(seed);
        TreeParam& tree_params = tree.getParams();

        // Add variation to tree parameters
        uint32_t variation_seed = tea<4>(seed, 555);
        
        // Random shape variation (0-8)
        tree_params.shape = rndInt(variation_seed, 0, 8);
        
        // Base parameters with variation
        tree_params.g_scale = 80.0f + rnd(variation_seed) * 40.0f;  // 80-120
        tree_params.g_scale_v = 2.0f + rnd(variation_seed) * 2.0f;  // 1.5-2.5
        tree_params.levels = 3;  // 2 or 3 levels
        tree_params.ratio = 0.025f + rnd(variation_seed) * 0.015f;  // 0.025-0.04 trunk thickness
        
        // Branch length variation
        tree_params.length[1] = 0.2f + rnd(variation_seed) * 0.3f;  // 0.1-0.3
        tree_params.length[2] = 0.3f + rnd(variation_seed) * 0.4f;  // 0.3-0.7
        tree_params.length[3] = 0.3f + rnd(variation_seed) * 0.2f;  // 0.3-0.5
        
        // Branch count variation
        tree_params.branches[0] = 1;  // Always single trunk
        tree_params.branches[1] = rndInt(variation_seed, 20, 40);   // 10-40
        tree_params.branches[2] = rndInt(variation_seed, 5, 30);   // 5-30
        tree_params.branches[3] = rndInt(variation_seed, 5, 10);   // 5-10

        tree_params.curve_res = {5,4,3,2};
        
        // Leaf parameters with variation
        tree_params.leaves = rndInt(variation_seed, 20, 35);  // 20-35
        tree_params.leaf_blos_num = rndInt(variation_seed, 20, 35);
        tree_params.leaf_scale = 0.008f + rnd(variation_seed) * 0.004f;  // 0.008-0.012
        
        float x_power = -2.0f + rnd(variation_seed) * 4.0f;  // -1.0 to 1.0
        float z_power = -2.0f + rnd(variation_seed) * 4.0f;  // -1.0 to 1.0
        float y_power = -5.0f + rnd(variation_seed) * 5.0f;
        tree_params.tropism = {x_power, y_power, z_power};
        
        // Decide leaf texture variation (single color or mixed)
        uint32_t leaf_seed = tea<4>(seed, 888);
        vector<int> leaf_texture_ids;
        
        float single_color_chance = 0.3f;  // 30% chance of single-color tree
        if (rnd(leaf_seed) < single_color_chance) {
            // Single-color leaves: pick one texture
            int tex_id = rndInt(leaf_seed, 0, leafs.size() - 1);
            leaf_texture_ids.push_back(tex_id);
        } else {
            // Mixed leaves: use 2-3 random textures
            int num_textures = rndInt(leaf_seed, 2, 3);
            for (int t = 0; t < num_textures; t++) {
                int tex_id = rndInt(leaf_seed, 0, leafs.size() - 1);
                leaf_texture_ids.push_back(tex_id);
            }
        }

        float pos_x = tree_positions_2d[i].x();
        float pos_z = tree_positions_2d[i].y();
        
        // Get terrain height at this position
        float terrain_height = getTerrainHeight(pos_x, pos_z);
        
        Vec3f tree_pos(pos_x, terrain_height + tree_height_offset, pos_z);
            auto [tree_mesh, leaf_mesh] = buildTreeMesh(seed, tree, tree_params, leaf_texture_ids);
            
            // Select bark type randomly for variety
            uint32_t bark_seed = tea<4>(seed, 777);
            int bark_type = rndInt(bark_seed, 0, 2);  // 0=rough, 1=aged, 2=smooth
            
            shared_ptr<FloatBitmapTexture> selected_bark;
            shared_ptr<Texture> bark_base_texture;
            string bark_type_name;
            switch(bark_type) {
                case 0:
                    selected_bark = rough_bark_texture;
                    bark_base_texture = procedural_wooden_texture;
                    bark_type_name = "rough";
                    break;
                case 1:
                    selected_bark = aged_bark_texture;
                    bark_base_texture = procedural_wooden_texture;
                    bark_type_name = "aged";
                    break;
                case 2:
                    selected_bark = nullptr;  // No bumpmap for smooth
                    bark_base_texture = smooth_bark_texture;  // Use color texture
                    bark_type_name = "smooth";
                    break;
            }
            
            // Create tree material
            auto tree_material = make_shared<Diffuse>(diffuse_id, bark_base_texture, false);
            if (selected_bark) {
                tree_material->setBumpmap(selected_bark);  // Only rough/aged have bumpmaps
            }
            
            string tree_name = "tree_" + to_string(i);
            m_scene.addObject(tree_name, tree_mesh, tree_material, mesh_prgs, Matrix4f::translate(tree_pos));

            
            // Add leaves if generated
            if (leaf_mesh) {
                string leaf_name = "leaves_" + to_string(i);
                
                // Create vector of leaf materials (one per texture)
                vector<shared_ptr<Material>> leaf_materials;
                for (auto& leaf_tex : leafs) {
                    leaf_tex->copyToDevice();
                    auto leaf_diffuse = make_shared<Diffuse>(diffuse_id, leaf_tex, true);
                    leaf_diffuse->setOpacityTexture(leaf_tex);
                    leaf_materials.push_back(leaf_diffuse);
                }
                
                m_scene.addObject(leaf_name, leaf_mesh, leaf_materials, mesh_prgs, Matrix4f::translate(tree_pos));
            }
    }

    // Generate Voronoi rocks at random positions on terrain
    // NOTE: Each rock must be generated separately because:
    // 1. Voronoi generation is randomized, creating unique shapes each time
    // 2. Y-normalization must be applied per-rock to ensure bottom at Y=0
    // 3. Reusing a single mesh would cause inconsistent ground placement
    
    const int num_rocks = 100;  // Increased from 5 to 15
    for (int i = 0; i < num_rocks; i++) {
        // Create unique seed for this rock (independent for shape and position)
        uint32_t rock_seed = tea<4>(seed, i * 137);  // Use prime number for better distribution
        
        // Create unique parameters for each rock
        VoronoiRockParams rock_params = createDefaultRockFieldParams();
        rock_params.seed_count = 1;             // Generate SINGLE rock (not 30!)
        rock_params.field_size = 10.0f;         // Small field for single rock
        rock_params.y_position = 0.0f;          // Base Y position (rock bottom will be at 0)
        rock_params.min_rock_vertices = 20;     // Minimum vertices per rock
        rock_params.max_rock_vertices = 40;     // Maximum vertices per rock
        rock_params.rock_roughness = 0.3f;      // Surface roughness variation
        rock_params.rock_height_variation = 0.4f; // Y-axis compression variation
        rock_params.rock_base_size = 1.0f;      // Base size (independent of field_size!)
        rock_params.random_seed = rock_seed;    // Unique seed for rock generation
        
        // Generate unique rock mesh for this position
        auto rock_mesh = buildVoronoiRockMesh(rock_params);
        
        // Random position within terrain bounds (80% of terrain to ensure no overflow)
        // Use separate seed for position to avoid correlation with rock shape
        uint32_t pos_seed = tea<4>(seed, i * 239 + 1);  // Different prime for X
        float rock_x = (rnd(pos_seed) - 0.5f) * terrain_params.terrain_size * 0.8f;
        
        pos_seed = tea<4>(seed, i * 239 + 2);  // Different offset for Z
        float rock_z = (rnd(pos_seed) - 0.5f) * terrain_params.terrain_size * 0.8f;
        
        // Get terrain height at this position
        float rock_height = getTerrainHeight(rock_x, rock_z);
        
        // Random scale for variety (medium-sized rocks)
        pos_seed = tea<4>(seed, i * 239 + 3);  // Different offset for scale
        float rock_scale = 5.0f + rnd(pos_seed) * 5.0f;  // 5-10 scale
        
        // Rock mesh is centered at origin (Y=0 is at rock center, not bottom)
        // Transform: Scale first, then translate to world position
        // Rock center will be placed at terrain height
        
        auto rock_diffuse = make_shared<Diffuse>(diffuse_id, grey_texture);
        rock_diffuse->setBumpmap(rock_bumpmap_texture);
        
        string rock_name = "rock_" + to_string(i);
        Matrix4f rock_transform = Matrix4f::translate(Vec3f(rock_x, rock_height, rock_z)) * Matrix4f::scale(rock_scale);
        m_scene.addObject(rock_name, rock_mesh, rock_diffuse, mesh_prgs, rock_transform);
    }

    auto bunny_white = make_shared<Diffuse>(diffuse_id, white_texture);
    auto bunny_glass = make_shared<Dielectric>(dielectric_id, white_texture, 1.5f, 0.01f);
    auto bunny_disney = make_shared<Disney>(
        disney_id, 
        winered_checker_texture, 
        /* subsurface = */ 0.2f, 
        /* metallic = */ 0.9f, 
        /* specular = */ 0.5f, 
        /* specular_tint = */ 0.5f, 
        /* roughness = */ 0.3f, 
        /* anisotropic = */ 0.5f, 
        /* sheen = */ 0.0f, 
        /* sheen_tint = */ 0.0f,
        /* clearcoat = */ 0.0f,
        /* clearcoat_gloss = */ 0.0f
    );

    auto bunny_mesh = make_shared<TriangleMesh>("uv_bunny.obj");
    // Bunny 
    {
        m_bunny1_scale = 150.0f;
        Matrix4f bunny_transform = Matrix4f::translate(m_bunny1_pos) * Matrix4f::scale(m_bunny1_scale);
        m_scene.addObject("bunny1", bunny_mesh, bunny_white, mesh_prgs, bunny_transform);
    }

    {
        m_bunny2_scale = 180.0f;
        Matrix4f bunny_transform = Matrix4f::translate(m_bunny2_pos) * Matrix4f::rotate(-math::pi / 6.0f, Vec3f(0,1,0)) * Matrix4f::scale(m_bunny2_scale);
        m_scene.addObject("bunny2", bunny_mesh, bunny_disney, mesh_prgs, bunny_transform);
    }

    {
        m_bunny3_scale = 120.0f;
        Matrix4f bunny_transform = Matrix4f::translate(m_bunny3_pos) * Matrix4f::rotate(-math::two_pi / 3.0f, Vec3f(0, 1, 0)) * Matrix4f::scale(m_bunny3_scale);
        m_scene.addObject("bunny3", bunny_mesh, bunny_glass, mesh_prgs, bunny_transform);
    }

    // Area lights
    auto addLight = [&](const string& name, shared_ptr<Shape> shape, shared_ptr<AreaEmitter> emitter, array<ProgramGroup, NRay>& prgs, const Matrix4f& transform, uint32_t sample_id, uint32_t pdf_id) {
        emitter->copyToDevice();
        shape->copyToDevice();
        m_scene.addLight(name, shape, emitter, prgs, transform);
        AreaEmitterInfo light_info = {
            .shape_data = shape->devicePtr(),
            .surface_info = emitter->surfaceInfoDevicePtr(),
            .objToWorld = transform,
            .worldToObj = transform.inverse(),
            .sample_id = sample_id,
            .pdf_id = pdf_id
        };
        m_light_infos.insert(make_pair(name, light_info));
    };

    m_light_ease = EaseType::Linear;
    Vec3f first_light_pos(-200.0f, 15.5f, 53.8f);
    Vec3f final_light_pos = (m_bunny1_pos + m_bunny2_pos + m_bunny3_pos) / 3.0f;
    final_light_pos.y() += 40.0f;

    Vec3f second_pos = lerp(first_light_pos, final_light_pos, 2.0f / 7.5f);
    Vec3f third_pos = lerp(first_light_pos, final_light_pos, 3.5f / 7.5f);
    third_pos.y() += 15.0f;
    
    m_light_points = {
        {first_light_pos, 0.0f},
        {second_pos, VIDEO_LENGTH * 0.3f},
        {third_pos, VIDEO_LENGTH * 0.5f},
        {final_light_pos, VIDEO_LENGTH * 0.9f}
    };

    addLight("light1", 
      make_shared<Sphere>(final_light_pos, 3.0f), 
      make_shared<AreaEmitter>(area_emitter_id, 
          make_shared<ConstantTexture>(Vec3f(0.5f, 0.5f, 0.9f), constant_id),
          10.0f), 
      sphere_prgs, 
      Matrix4f::identity(), 
      sphere_light_sample_id, sphere_light_pdf_id);  // Use sphere light sampling callable

    // Copy light info to GPU
    copyAreaEmitterToDevice();

    CUDA_CHECK(cudaStreamCreate(&m_stream));
    m_scene.copyDataToDevice();
    m_scene.buildAccel(m_ctx, m_stream);
    m_scene.buildSBT();
    m_ppl.create(m_ctx);

    m_params.handle = m_scene.accelHandle();

#if SUBMISSION
    
    std::cout << "Starting rendering... (Target: " << n_frame << " frames)\n";
    double total_render_time = 0.0;
    
    int frame = 0;
    while (frame < n_frame && !time_limit_exceeded.load()) {

        std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();

        // Check time limit before starting new frame
        if (time_limit_exceeded.load()) {
            break;
        }

        is_camera_updated = true;
        handleCameraUpdate();

        for (uint32_t iter = 0; iter < NUM_ITER; iter++) {
            // Check time limit during iterations
            if (time_limit_exceeded.load()) {
                break;
            }

            m_params.samples_per_launch = SPP_PER_LAUNCH;

            m_scene.launchRay(m_ctx, m_ppl, m_params, m_stream, m_bitmap.width(), m_bitmap.height(), 1);
            CUDA_SYNC_CHECK();

            if (enable_firefly_filter) {
                const int width = m_bitmap.width();
                const int height = m_bitmap.height();

                FireflyFilterParams firefly_params;
                firefly_params.outlier_ratio = 2.5f;     // 2.5x brighter than neighbors = outlier
                firefly_params.min_luminance = 0.01f;    // Minimum luminance to consider (avoid dark areas)

                launchFireflyFilterKernel(
                    m_params.float_result_buffer,
                    d_firefly_temp,  // Use temp buffer as output
                    width,
                    height,
                    firefly_params,
                    m_stream
                );

                // Copy filtered result back to float_result_buffer
                CUDA_CHECK(cudaMemcpyAsync(
                    m_params.float_result_buffer,
                    d_firefly_temp,
                    width * height * sizeof(Vec4f),
                    cudaMemcpyDeviceToDevice,
                    m_stream
                ));
            }
            
            m_params.frame = iter;
        }        // Bloom
        BloomParams bloom_params;
        bloom_params.threshold = bloom_threshold;
        bloom_params.intensity = bloom_intensity;
        bloom_params.blur_radius = bloom_radius;
        bloom_params.sigma = bloom_sigma;

        applyBloomEffect(
            m_params.float_result_buffer,
            (Vec4f*)m_bloom_bitmap.deviceData(),
            d_bloom_temp1,
            d_bloom_temp2,
            width,
            height,
            bloom_params,
            m_stream
        );

        auto render_time = system_clock::now() - start_time;
        total_render_time += (double)duration_cast<milliseconds>(render_time).count() / 1000.0;
        
        // Output rendered image (only if not interrupted)
        if (!time_limit_exceeded.load()) {
            m_bloom_bitmap.copyFromDevice();
            string filename = format("{:03d}.png", frame);
            filesystem::path filepath = pgPathJoin(pgGetExecutableDir(), filename);
            m_bloom_bitmap.write(filepath);

            double elapsed_seconds = (double)duration_cast<milliseconds>(system_clock::now() - start_time).count() / 1000.0;
            std::cout << "\rElapsed: " << std::fixed << std::setprecision(1) 
                      << elapsed_seconds << "s / " << TIME_LIMIT << "s | Frame: " 
                      << std::setw(3) << std::setfill('0') << frame << "/" << n_frame << std::flush;

            frame++;
            m_frame_time += m_interval;
        }

        // Update scene 
        //m_scene.updateSBT(+(SBTRecordType::Hitgroup));

        // Find two keypoints to interpolate
        bool keypoint_found = false;
        Vec3f camera_pos = getValueFromKeypoints(m_frame_time, 0, VIDEO_LENGTH, m_cam_points, m_camera_ease);
        Vec3f camera_look = getValueFromKeypoints(m_frame_time, 0, VIDEO_LENGTH, m_look_points, m_camera_ease);
        m_scene.camera()->setOrigin(camera_pos);
        m_scene.camera()->setLookat(camera_look);

        // Update light position
        auto light1 = m_scene.getLight("light1");
        // Get sphere pointer from abstract shape class
        auto light1_sphere = dynamic_pointer_cast<Sphere>(light1->shape);
        Vec3f light_pos = getValueFromKeypointsBezier(m_frame_time, 0, VIDEO_LENGTH, m_light_points, 0.3f);
        light1_sphere->setCenter(light_pos);
        light1_sphere->copyToDevice();
        m_scene.updateLightGAS("light1", m_ctx, m_stream);
        copyAreaEmitterToDevice();

        m_scene.updateAccel(m_ctx, m_stream);

        is_camera_updated = true;
    }

    std::cout << "\nRendering finished. Total render time: " 
              << std::fixed << std::setprecision(2) << total_render_time << " seconds.\n"
              << "Average time per frame: "
              << std::fixed << std::setprecision(2) << (total_render_time / frame) << " seconds.\n";
    
    rendering_complete.store(true);  // Signal watchdog thread
    std::cout << "\n";  // New line after completion
    
    if (time_limit_exceeded.load()) {
        pgLog(format("Rendering stopped due to time limit. {} frames completed.", frame));
    } else {
        pgLog("Rendering completed successfully!");
    }
    
    pgExit();
#else
    // GUI settings
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(pgGetCurrentWindow()->windowPtr(), true);
    ImGui_ImplOpenGL3_Init("#version 330");
#endif

}

// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();

#if USE_SVGF
    // Temporal Anti-Aliasing: Apply sub-pixel jitter
    Vec2f jitter = getJitter(m_taa_frame_index);
    m_taa_frame_index++;
    
    // Update view-projection matrices for motion vector calculation
    Matrix4f curr_vp = m_scene.camera()->getViewProjectionMatrix();
    
    // Convert Matrix4f to POD MatrixData (first 12 elements)
    MatrixData curr_vp_data;
    MatrixData prev_vp_data;
    for (int i = 0; i < 12; i++) {
        curr_vp_data.data[i] = curr_vp[i];
        prev_vp_data.data[i] = m_prev_vp_matrix[i];
    }
    
    m_params.prev_view_projection = prev_vp_data;
    m_params.curr_view_projection = curr_vp_data;
#else
#endif

#if INTERACTIVE
    m_scene.launchRay(m_ctx, m_ppl, m_params, m_stream, m_bitmap.width(), m_bitmap.height(), 1);
    CUDA_SYNC_CHECK();

    // Firefly filtering (before bloom to prevent fireflies from spreading)
    if (enable_firefly_filter) {
        const int width = m_bitmap.width();
        const int height = m_bitmap.height();

        FireflyFilterParams firefly_params;
        firefly_params.outlier_ratio = 2.5f;     // 2.5x brighter than neighbors = outlier
        firefly_params.min_luminance = 0.01f;    // Minimum luminance to consider (avoid dark areas)

        launchFireflyFilterKernel(
            m_params.float_result_buffer,
            d_bloom_temp1,  // Use temp buffer as output
            width,
            height,
            firefly_params,
            m_stream
        );
        
        // Copy filtered result back to float_result_buffer
        CUDA_CHECK(cudaMemcpyAsync(
            m_params.float_result_buffer,
            d_bloom_temp1,
            width * height * sizeof(Vec4f),
            cudaMemcpyDeviceToDevice,
            m_stream
        ));
    }
#else

    for (uint32_t iter = 0; iter < NUM_ITER; iter++) {
        m_params.samples_per_launch = SPP_PER_LAUNCH;

        m_scene.launchRay(m_ctx, m_ppl, m_params, m_stream, m_bitmap.width(), m_bitmap.height(), 1);
        CUDA_SYNC_CHECK();

        // Firefly filtering (before bloom to prevent fireflies from spreading)
        const int width = m_bitmap.width();
        const int height = m_bitmap.height();

        FireflyFilterParams firefly_params;
        firefly_params.outlier_ratio = 2.5f;     // 2.5x brighter than neighbors = outlier
        firefly_params.min_luminance = 0.01f;    // Minimum luminance to consider (avoid dark areas)

        launchFireflyFilterKernel(
            m_params.float_result_buffer,
            d_firefly_temp,  // Use temp buffer as output
            width,
            height,
            firefly_params,
            m_stream
        );
        
        // Copy filtered result back to float_result_buffer
        CUDA_CHECK(cudaMemcpyAsync(
            m_params.float_result_buffer,
            d_firefly_temp,
            width * height * sizeof(Vec4f),
            cudaMemcpyDeviceToDevice,
            m_stream
        ));

        // Bloom
        BloomParams bloom_params;
        bloom_params.threshold = bloom_threshold;
        bloom_params.intensity = bloom_intensity;
        bloom_params.blur_radius = bloom_radius;
        bloom_params.sigma = bloom_sigma;

        applyBloomEffect(
            m_params.float_result_buffer,
            m_params.float_result_buffer,
            d_bloom_temp1,
            d_bloom_temp2,
            width,
            height,
            bloom_params,
            m_stream
        );

        m_params.frame = iter;
    }

    // Find two keypoints to interpolate
    float min_t = 0.0f; float max_t = VIDEO_LENGTH;
    bool keypoint_found = false;
    for (int i = 0; i < m_cam_points.size() - 1; i++) {
        min_t = fminf(min_t, m_cam_points[i].t);
        max_t = fmaxf(max_t, m_cam_points[i].t);
        if (m_cam_points[i].t <= m_frame_time && m_frame_time < m_cam_points[i + 1].t) {
            Vec3f pos = KeyPoint<Vec3f>::ease(m_cam_points[i], m_cam_points[i + 1], m_frame_time, m_camera_ease);
            Vec3f look = KeyPoint<Vec3f>::ease(m_look_points[i], m_look_points[i + 1], m_frame_time, m_camera_ease);
            m_scene.camera()->setOrigin(pos);
            m_scene.camera()->setLookat(look);
            keypoint_found = true;
            break;
        }
    }

    if (!keypoint_found) {
        min_t = fminf(min_t, m_cam_points[m_cam_points.size() - 1].t);
        max_t = fmaxf(max_t, m_cam_points[m_cam_points.size() - 1].t);

        if (m_frame_time < min_t) {
            m_scene.camera()->setOrigin(m_cam_points[0].value);
            m_scene.camera()->setLookat(m_look_points[0].value);
        }
        else if (m_frame_time > max_t) {
            m_scene.camera()->setOrigin(m_cam_points[m_cam_points.size() - 1].value);
            m_scene.camera()->setLookat(m_look_points[m_look_points.size() - 1].value);
        }
    }

    // Update light position
    auto light1 = m_scene.getLight("light1");
    // Get sphere pointer from abstract shape class
    auto light1_sphere = dynamic_pointer_cast<Sphere>(light1->shape);
    light1_sphere->setCenter(Vec3f(-30.0f + 100.0f * sinf(0.1f * m_frame_time), 80.0f, -30.0f + 100.0f * cosf(0.1f * m_frame_time)));
    light1_sphere->copyToDevice();
    m_scene.updateLightGAS("light1", m_ctx, m_stream);
    copyAreaEmitterToDevice();

    is_camera_updated = true;
    m_scene.updateAccel(m_ctx, m_stream);
    m_frame_time += m_interval;
#endif

#if DENOISE
    m_float_bitmap.copyFromDevice();
    m_normal_bitmap.copyFromDevice();
    m_albedo_bitmap.copyFromDevice();

    m_denoise_data.color = m_float_bitmap.deviceData();
    m_denoise_data.normal = m_normal_bitmap.deviceData();
    m_denoise_data.albedo = m_albedo_bitmap.deviceData();

    m_denoiser.update(m_denoise_data);

    m_denoiser.run();

    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    CUDA_SYNC_CHECK();

    m_denoiser.copyFromDevice();
#endif

#if USE_SVGF
    // Apply SVGF filter
    // Input: m_params.float_result_buffer (tone-mapped HDR with bloom)
    // Output: m_svgf_output (denoised)
    // G-Buffer: Uses existing normal/albedo buffers directly (zero-copy!)
    m_svgf.filter(
        m_params.float_result_buffer,
        m_svgf_gbuffer,
        (Vec4f*)m_svgf_output.deviceData(),
        m_stream
    );
    
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    CUDA_SYNC_CHECK();
    
    // Copy denoised result back to float_result_buffer for display
    CUDA_CHECK(cudaMemcpyAsync(
        m_params.float_result_buffer,
        m_svgf_output.deviceData(),
        m_bitmap.width() * m_bitmap.height() * sizeof(Vec4f),
        cudaMemcpyDeviceToDevice,
        m_stream
    ));
#endif

    // Bloom
    if (enable_bloom) {
        const int width = m_bitmap.width();
        const int height = m_bitmap.height();

        BloomParams bloom_params;
        bloom_params.threshold = bloom_threshold;
        bloom_params.intensity = bloom_intensity;
        bloom_params.blur_radius = bloom_radius;
        bloom_params.sigma = bloom_sigma;

        applyBloomEffect(
            m_params.float_result_buffer,
            (Vec4f*)m_bloom_bitmap.deviceData(),
            d_bloom_temp1,
            d_bloom_temp2,
            width,
            height,
            bloom_params,
            m_stream
        );
    }

    m_params.frame++;

#if USE_SVGF
    // Store current VP matrix for next frame
    m_prev_vp_matrix = m_scene.camera()->getViewProjectionMatrix();
#endif
}
#ifdef ENABLE_AVG_LUM_DEBUG
    // Debug: copy accum buffer to host and print average luminance
    {
        const size_t n = static_cast<size_t>(m_bitmap.width()) * static_cast<size_t>(m_bitmap.height());
        Vec4f* host_accum = (Vec4f*)malloc(n * sizeof(Vec4f));
        if (host_accum) {
            CUDA_CHECK(cudaMemcpy(host_accum, m_params.accum_buffer, n * sizeof(Vec4f), cudaMemcpyDeviceToHost));
            double sumL = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const Vec3f c(host_accum[i]);
                const double lum = 0.212671 * c.x() + 0.715160 * c.y() + 0.072169 * c.z();
                sumL += lum;
            }
            double avgL = sumL / static_cast<double>(n);
            printf("[DEBUG] Avg accum luminance (frame %d): %f\n", m_params.frame, avgL);
            free(host_accum);
        }
    }
#endif

// ------------------------------------------------------------------
void App::draw()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("RTCAMP11");

    auto camera = m_scene.camera();
    bool state_changed = false;
    if (ImGui::Button("Reset movie"))
        resetMovie();
    ImGui::Text("Camera info:");
    ImGui::Text("  Origin: (%.2f, %.2f, %.2f)", camera->origin().x(), camera->origin().y(), camera->origin().z());
    ImGui::Text("  Lookat: (%.2f, %.2f, %.2f)", camera->lookat().x(), camera->lookat().y(), camera->lookat().z());

    ImGui::Text("Frame rate: %.3f ms/frame (%.2f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Subframe index: %d", m_params.frame);

    ImGui::Separator();
    ImGui::Text("General parameters");
    state_changed |= ImGui::SliderFloat("White", &m_params.white, 1.0f, 30.0f, "%.2f");

    // Camera position and lookat controls by sliders
    ImGui::Separator();
    ImGui::Text("Camera Controls (Slider)");
    Vec3f cam_origin = camera->origin();
    Vec3f cam_lookat = camera->lookat();
    state_changed |= ImGui::SliderFloat3("Origin", &cam_origin[0], -500.0f, 500.0f, "%.2f");
    state_changed |= ImGui::SliderFloat3("Lookat", &cam_lookat[0], -500.0f, 500.0f, "%.2f");
    if (state_changed) {
        is_camera_updated = true;
        m_scene.camera()->setOrigin(cam_origin);
        m_scene.camera()->setLookat(cam_lookat);
    }

    ImGui::Text("Camera Controls (Input)");
    state_changed |= ImGui::InputFloat3("Origin Input", &cam_origin[0], "%.2f");
    state_changed |= ImGui::InputFloat3("Lookat Input", &cam_lookat[0], "%.2f");
    if (state_changed) {
        is_camera_updated = true;
        m_scene.camera()->setOrigin(cam_origin);
        m_scene.camera()->setLookat(cam_lookat);
    }

    //ImGui::Text("Bunny1 Control");
    //state_changed |= ImGui::InputFloat3("Bunny1 Position", &m_bunny1_pos[0], "%.2f");
    //state_changed |= ImGui::SliderFloat("Bunny1 Scale", &m_bunny1_scale, 10.0f, 300.0f, "%.2f");
    //if (state_changed) {
    //    m_scene.updateObjectTransform("bunny1", Matrix4f::translate(m_bunny1_pos) * Matrix4f::scale(m_bunny1_scale));
    //    m_scene.updateAccel(m_ctx, m_stream);
    //}

    //ImGui::Text("Bunny2 Control");
    //state_changed |= ImGui::InputFloat3("Bunny2 Position", &m_bunny2_pos[0], "%.2f");
    //state_changed |= ImGui::SliderFloat("Bunny2 Scale", &m_bunny2_scale, 10.0f, 300.0f, "%.2f");
    //if (state_changed) {
    //    m_scene.updateObjectTransform("bunny2", Matrix4f::translate(m_bunny2_pos) * Matrix4f::rotate(-math::pi / 6.0f, Vec3f(0, 1, 0)) * Matrix4f::scale(m_bunny2_scale));
    //    m_scene.updateAccel(m_ctx, m_stream);
    //}

    //ImGui::Text("Bunny3 Control");
    //state_changed |= ImGui::InputFloat3("Bunny3 Position", &m_bunny3_pos[0], "%.2f");
    //state_changed |= ImGui::SliderFloat("Bunny3 Scale", &m_bunny3_scale, 10.0f, 300.0f, "%.2f");
    //if (state_changed) {
    //    m_scene.updateObjectTransform("bunny3", Matrix4f::translate(m_bunny3_pos) * Matrix4f::rotate(-math::two_pi / 3.0f, Vec3f(0, 1, 0)) * Matrix4f::scale(m_bunny3_scale));
    //    m_scene.updateAccel(m_ctx, m_stream);
    //}

    ImGui::Text("Light Control");
    auto light1 = m_scene.getLight("light1");
    // Get sphere pointer from abstract shape class
    auto light1_sphere = dynamic_pointer_cast<Sphere>(light1->shape);
    Vec3f center = light1_sphere->center();
    state_changed |= ImGui::SliderFloat3("Light Position", &center[0], -500.0f, 500.0f, "%.2f");
    if (state_changed) {
        light1_sphere->setCenter(center);
        light1_sphere->copyToDevice();
        m_scene.updateLightGAS("light1", m_ctx, m_stream);
        m_scene.updateAccel(m_ctx, m_stream);
        copyAreaEmitterToDevice();
    }

    ImGui::Separator();
    if (ImGui::Checkbox("Enable Firefly filter", &enable_firefly_filter)) {
        // Reset frame counter when toggling bloom
        initResultBufferOnDevice();
    }

    if (ImGui::Checkbox("Visualize G-Buffer", &enable_gbuffer)) {
        // Reset frame counter when toggling bloom
        initResultBufferOnDevice();
    }

    if (ImGui::Checkbox("Enable MIS", &m_params.enable_mis)) {
        initResultBufferOnDevice();
    }

    // Bloom effect controls
    ImGui::Separator();
    ImGui::Text("Bloom Effect");
    if (ImGui::Checkbox("Enable Bloom", &enable_bloom)) {
        // Reset frame counter when toggling bloom
        initResultBufferOnDevice();
    }
    
    if (enable_bloom) {
        state_changed |= ImGui::SliderFloat("Threshold", &bloom_threshold, 0.0f, 3.0f, "%.2f");
        state_changed |= ImGui::SliderFloat("Intensity", &bloom_intensity, 0.0f, 1.0f, "%.2f");
        state_changed |= ImGui::SliderInt("Blur Radius", &bloom_radius, 1, 20);
        state_changed |= ImGui::SliderFloat("Blur Sigma", &bloom_sigma, 1.0f, 20.0f, "%.1f");
    }

    if (state_changed) {
        // Reset frame counter when scene parameters change
        initResultBufferOnDevice();
    }

    ImGui::End();
    ImGui::Render();

    auto w = pgGetWidth();
    auto h = pgGetHeight();

    m_bloom_bitmap.copyFromDevice();
    m_float_bitmap.copyFromDevice();

    if (enable_gbuffer) {
        m_albedo_bitmap.copyFromDevice();
        m_normal_bitmap.copyFromDevice();
        m_uv_bitmap.copyFromDevice();

        //m_bloom_bitmap.draw(0, 0, w / 2, h / 2);
        m_albedo_bitmap.draw(0, 0, w / 2, h / 2);
        m_normal_bitmap.draw(0, h / 2, w / 2, h / 2);
        m_uv_bitmap.draw(w / 2, 0, w / 2, h / 2);
    }
    else {
        if (enable_bloom)
            m_bloom_bitmap.draw(0, 0, w, h);
        else
            m_float_bitmap.draw(0, 0, w, h);
    }
    
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// ------------------------------------------------------------------
void App::mousePressed(float x, float y, int button)
{
    
}

// ------------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    if (button == MouseButton::Middle) is_camera_updated = true;
}

// ------------------------------------------------------------------
void App::mouseReleased(float x, float y, int button)
{
    
}

// ------------------------------------------------------------------
void App::mouseMoved(float x, float y)
{
    
}

// ------------------------------------------------------------------
void App::mouseScrolled(float x, float y)
{
    is_camera_updated = true;
}

// ------------------------------------------------------------------
void App::keyPressed(int key)
{

}

// ------------------------------------------------------------------
void App::keyReleased(int key)
{
    if (key == Key::Up) {
        m_scene.camera()->setOrigin(m_scene.camera()->origin() + Vec3f(0, 1, 0));
        m_scene.camera()->setLookat(m_scene.camera()->lookat() + Vec3f(0, 1, 0));
        is_camera_updated = true;
    }
    else if (key == Key::Down) {
        m_scene.camera()->setOrigin(m_scene.camera()->origin() + Vec3f(0, -1, 0));
        m_scene.camera()->setLookat(m_scene.camera()->lookat() + Vec3f(0, -1, 0));
        is_camera_updated = true;
    }
    else if (key == Key::Left) {
        m_scene.camera()->setOrigin(m_scene.camera()->origin() + Vec3f(-1, 0, 0));
        m_scene.camera()->setLookat(m_scene.camera()->lookat() + Vec3f(-1, 0, 0));
        is_camera_updated = true;
    }
    else if (key == Key::Right) {
        m_scene.camera()->setOrigin(m_scene.camera()->origin() + Vec3f(1, 0, 0));
        m_scene.camera()->setLookat(m_scene.camera()->lookat() + Vec3f(1, 0, 0));
        is_camera_updated = true;
    }
    else if (key == Key::F) {
        m_scene.camera()->setOrigin(m_scene.camera()->origin() + m_scene.camera()->direction() * 1.0f);
        m_scene.camera()->setLookat(m_scene.camera()->lookat() + m_scene.camera()->direction() * 1.0f);
        is_camera_updated = true;
    }
    else if (key == Key::B) {
        m_scene.camera()->setOrigin(m_scene.camera()->origin() - m_scene.camera()->direction() * 1.0f);
        m_scene.camera()->setLookat(m_scene.camera()->lookat() - m_scene.camera()->direction() * 1.0f);
        is_camera_updated = true;
    }
    else if (key == Key::S) {
        if (enable_bloom) {
            m_bloom_bitmap.copyFromDevice();
            m_bloom_bitmap.write("frame.png");
        }
        else {
            m_float_bitmap.copyFromDevice();
            m_float_bitmap.write("frame.png");
        }
    }
}



