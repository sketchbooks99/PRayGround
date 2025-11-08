#include "app.h"
#include "textures.cuh"
#include <queue>

// ------------------------------------------------------------------
void App::initResultBufferOnDevice()
{
    m_params.frame = 0;
    m_bitmap.allocateDevicePtr();
    m_accum_buffer.allocateDevicePtr();

    m_params.result_buffer = (Vec4u*)m_bitmap.deviceData();
    m_params.accum_buffer = (Vec4f*)m_accum_buffer.deviceData();
}

// ------------------------------------------------------------------
void App::handleCameraUpdate()
{
    if (!is_camera_updated) return;

    is_camera_updated = false;

    m_scene.updateSBT(+(SBTRecordType::Raygen));

    initResultBufferOnDevice();
}

// ------------------------------------------------------------------
void App::setup()
{
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
    m_bitmap.allocateDevicePtr();
    m_accum_buffer.allocate(PixelFormat::RGBA, width, height);
    m_accum_buffer.allocateDevicePtr();

    // Configuration of launch parameters
    m_params.width = width;
    m_params.height = height;
    m_params.samples_per_launch = 4;
    m_params.frame = 0u;
    m_params.max_depth = 10u;
    m_params.result_buffer = (Vec4u*)m_bitmap.deviceData();
    m_params.accum_buffer = (Vec4f*)m_accum_buffer.deviceData();
    m_params.white = 5.0f;

    // Setup scene
    AppScene::AccelSettings accel_settings = {
        .allow_accel_compaction = true,
        .allow_accel_update = true
    };
    m_scene.setup(accel_settings);

    // Camera settings
    shared_ptr<Camera> camera = make_shared<Camera>();
    camera->setOrigin(0, 20, -100);
    camera->setLookat(0, 0, 0);
    camera->setUp(0, 1, 0);
    camera->setFov(40);
    camera->setAspect((float)width / height);
    camera->enableTracking(pgGetCurrentWindow());
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
    
    // Register environment map importance sampling callable functions
    auto envmap_sample_id = setupCallable("__direct_callable__sample_envmap", "");
    auto envmap_pdf_id = setupCallable("__direct_callable__pdf_envmap", "");
    
    // Bake procedural texture and build CDF for importance sampling
    StarNightTexture::Data star_night_data{
        .base_color = Vec4f(0.0f, 0.0f, 0.01f, 1.0f),
        .noise_data = RandomNoise::Data{
            .seed = 1234,
            .width = 4096, 
            .height = 4096, 
            .depth = 1
        },
        .star_threshold = 0.9995f,
        .star_intensity = 50.0f,
        .moon_dir = normalize(Vec3f(1.0f, 0.2f, -1.0f)),
        .moon_intensity = 300.0f
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
    
    printf("[Envmap] Importance sampling enabled (sample_id: %u, pdf_id: %u)", 
           envmap_sample_id, envmap_pdf_id);

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
        printf("[ERROR] Failed to bake rock texture!");
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
        printf("[ERROR] Failed to create bump texture from heightmap!");
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

    auto leaf1 = make_shared<BitmapTexture>("foliage_10.png", bitmap_id);
    auto leaf2 = make_shared<BitmapTexture>("foliage_15.png", bitmap_id);
    auto leaf3 = make_shared<BitmapTexture>("foliage_24.png", bitmap_id);
    auto leaf4 = make_shared<BitmapTexture>("foliage_52.png", bitmap_id);
    auto leaf5 = make_shared<BitmapTexture>("foliage_82.png", bitmap_id);
    auto leafs = vector<shared_ptr<BitmapTexture>>{ leaf1, leaf2, leaf3, leaf4, leaf5 };

    // Surfaces
    auto floor_diffuse = make_shared<Diffuse>(diffuse_id, checker_texture);
    auto tree_diffuse = make_shared<Diffuse>(diffuse_id, procedural_wooden_texture, false);

    // Generate terrain first (before placing trees/rocks)
    TerrainParams terrain_params;
    terrain_params.terrain_size = 500.0f;
    terrain_params.height_scale = 50.0f;
    auto terrain_result = buildTerrainMesh(terrain_params);
    auto terrain_mesh = terrain_result.mesh;
    auto terrain_heightmap = terrain_result.heightmap;
    auto terrain_texture = make_shared<FloatBitmapTexture>(terrain_heightmap, terrain_id);
    auto terrain_diffuse = make_shared<Diffuse>(diffuse_id, terrain_texture);
    
    // Terrain offset
    const Vec3f terrain_offset(0, -24, 0);
    m_scene.addObject("terrain", terrain_mesh, terrain_diffuse, mesh_prgs, Matrix4f::translate(terrain_offset));
    
    // Use actual min/max height from terrain generation
    const float terrain_min_height = terrain_result.min_height;
    const float terrain_max_height = terrain_result.max_height;
    
    printf("[Terrain] Using height range: %.2f to %.2f for object placement", 
           terrain_min_height, terrain_max_height);
    
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
    

    // Generate grid-based forest with slight perturbation (5x6 = 30 trees)
    const int grid_rows = 6;
    const int grid_cols = 5;
    const float grid_spacing = 60.0f;  // Distance between grid points
    const float perturbation = 8.0f;   // Random offset ±8 units
    
    vector<Vec2f> tree_positions_2d;
    
    printf("[Grid Forest] Placing %dx%d trees with spacing %.2f and perturbation ±%.2f", 
           grid_cols, grid_rows, grid_spacing, perturbation);
    
    // Calculate grid center offset
    float grid_width = (grid_cols - 1) * grid_spacing;
    float grid_depth = (grid_rows - 1) * grid_spacing;
    float offset_x = -grid_width * 0.5f;
    float offset_z = -grid_depth * 0.5f;
    
    // Generate grid with perturbation
    for (int row = 0; row < grid_rows; row++) {
        for (int col = 0; col < grid_cols; col++) {
            // Base grid position
            float base_x = offset_x + col * grid_spacing;
            float base_z = offset_z + row * grid_spacing;
            
            // Add random perturbation
            float perturb_x = (rnd(seed) - 0.5f) * perturbation * 2.0f;
            float perturb_z = (rnd(seed) - 0.5f) * perturbation * 2.0f;
            
            float pos_x = base_x + perturb_x;
            float pos_z = base_z + perturb_z;
            
            tree_positions_2d.push_back(Vec2f(pos_x, pos_z));
        }
    }
    
    printf("[Grid Forest] Successfully placed %zu trees in grid pattern", tree_positions_2d.size());
    
    float tree_height_offset = -3.0f;

    // Place trees at selected positions
    for (size_t i = 0; i < tree_positions_2d.size(); i++) {
        seed = tea<4>(i, 1);

        float pos_x = tree_positions_2d[i].x();
        float pos_z = tree_positions_2d[i].y();
        
        // Get terrain height at this position
        float terrain_height = getTerrainHeight(pos_x, pos_z);
        
        Vec3f tree_pos(pos_x, terrain_height + tree_height_offset, pos_z);
            
            // === OLD IMPLEMENTATION (comment out to test new API) ===
            /*
            // Randomly select tree type
            float tree_type = rnd(seed);
            ProceduralTreeData tree_data;

            float leaf_density = 10.0f + rnd(seed) * 10.0f - 9.0f;
            
            if (tree_type < 0.33f) {
                // Normal tree (deciduous style)
                tree_data.gravity = 0.1f + rnd(seed) * 0.2f;
                tree_data.coverage = 0.4f + rnd(seed) * 0.4f;
                tree_data.vertically = 2.0f + rnd(seed) * 1.0f;
                tree_data.twist = rnd(seed) * 0.5f;
                tree_data.trunk_mode = false;
                tree_data.scale = 40.0f + rnd(seed) * 20.0f;
                tree_data.depth = 4;
                tree_data.radius = 2.5f + rnd(seed) * 1.5f;
                tree_data.has_leaves = true;
                tree_data.leaf_density = leaf_density;
                tree_data.leaf_start_gen = 3;  // Leaves only on fine branches (gen 3+)
                tree_data.leaf_size = 1.0f + rnd(seed) * 0.5f;
            }
            else if (tree_type < 0.66f) {
                // Pine tree (coniferous style)
                tree_data.gravity = 0.05f + rnd(seed) * 0.1f;
                tree_data.coverage = 0.3f + rnd(seed) * 0.3f;
                tree_data.vertically = 0.4f + rnd(seed) * 0.3f;
                tree_data.twist = 0.0f;
                tree_data.trunk_mode = true;
                tree_data.scale = 50.0f + rnd(seed) * 20.0f;
                tree_data.depth = 4 + (rnd(seed) > 0.5f ? 1 : 0);
                tree_data.radius = 1.8f + rnd(seed) * 1.0f;
                tree_data.has_leaves = true;
                tree_data.leaf_density = leaf_density;
                tree_data.leaf_start_gen = 3;  // Leaves only on fine branches (gen 3+)
                tree_data.leaf_size = 0.5f + rnd(seed) * 1.0f;
            }
            else {
                // Spiral/exotic tree
                tree_data.gravity = 0.3f + rnd(seed) * 0.4f;
                tree_data.coverage = 1.5f + rnd(seed) * 1.0f;
                tree_data.vertically = 0.8f + rnd(seed) * 0.5f;
                tree_data.twist = 2.0f + rnd(seed) * 2.0f;
                tree_data.trunk_mode = rnd(seed) > 0.5f;
                tree_data.scale = 35.0f + rnd(seed) * 25.0f;
                tree_data.depth = 4 + (rnd(seed) > 0.3f ? 1 : 0);
                tree_data.radius = 4.0f + rnd(seed) * 3.0f;
                tree_data.has_leaves = true;
                tree_data.leaf_density = leaf_density;
                tree_data.leaf_start_gen = 3;  // Leaves only on fine branches (gen 3+)
                tree_data.leaf_size = 1.0f + rnd(seed) * 1.0f;
            }
            
            auto [tree_mesh, leaf_mesh] = buildTreeMesh(tree_data, seed, leafs);
            */
            // === END OLD IMPLEMENTATION ===
            
            // === NEW TREE API IMPLEMENTATION (for testing) ===
            auto [tree_mesh, leaf_mesh] = buildTreeMeshWithAPI(seed);
            // === END NEW IMPLEMENTATION ===
            
            string tree_name = "tree_" + to_string(i);
            m_scene.addObject(tree_name, tree_mesh, tree_diffuse, mesh_prgs, Matrix4f::translate(tree_pos));
            
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
    // First, create a single rock mesh to reuse (with min_y offset info)
    auto rock_mesh = buildVoronoiRockMesh();
    
    // Calculate rock's Y-range to place center at terrain level
    float rock_min_y = std::numeric_limits<float>::max();
    float rock_max_y = std::numeric_limits<float>::lowest();
    for (const auto& v : rock_mesh->vertices()) {
        if (v.y() < rock_min_y) rock_min_y = v.y();
        if (v.y() > rock_max_y) rock_max_y = v.y();
    }
    float rock_center_y = (rock_min_y + rock_max_y) * 0.5f;
    printf("[Rock Placement] Rock Y-range: %.2f to %.2f, center: %.2f", rock_min_y, rock_max_y, rock_center_y);
    
    const int num_rocks = 5;
    for (int i = 0; i < num_rocks; i++) {
        // Random position within terrain bounds (60% of terrain to ensure no overflow)
        float rock_x = (rnd(seed) - 0.5f) * terrain_params.terrain_size * 0.6f;
        float rock_z = (rnd(seed) - 0.5f) * terrain_params.terrain_size * 0.6f;
        
        // Get terrain height at this position
        float rock_height = getTerrainHeight(rock_x, rock_z);
        
        // Random scale for variety (medium-sized rocks)
        float rock_scale = 5.0f + rnd(seed) * 5.0f;  // 5-10 scale
        
        // Offset so rock's center (Y-axis) aligns with terrain surface
        // After scaling, center_y becomes center_y * rock_scale
        float rock_y_offset = -rock_center_y * rock_scale;
        
        printf("[Rock %d] Position: (%.2f, %.2f, %.2f), scale: %.2f, y_offset: %.2f", 
               i, rock_x, rock_height + rock_y_offset, rock_z, rock_scale, rock_y_offset);
        
        auto rock_diffuse = make_shared<Diffuse>(diffuse_id, grey_texture);
        rock_diffuse->setBumpmap(rock_bumpmap_texture);
        
        string rock_name = "rock_" + to_string(i);
        Matrix4f rock_transform = Matrix4f::translate(Vec3f(rock_x, rock_height + rock_y_offset, rock_z)) * Matrix4f::scale(rock_scale);
        m_scene.addObject(rock_name, rock_mesh, rock_diffuse, mesh_prgs, rock_transform);
    }

    // Area lights
    vector<AreaEmitterInfo> light_infos;
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
        light_infos.push_back(light_info);
    };

    addLight("light1", 
       make_shared<Sphere>(Vec3f(-30, 80, -30), 2.0f), 
       make_shared<AreaEmitter>(area_emitter_id, 
           make_shared<ConstantTexture>(Vec3f(0.9f, 0.85f, 0.7f), constant_id),
           300.0f), 
       sphere_prgs, 
       Matrix4f::identity(), 
       sphere_light_sample_id, sphere_light_pdf_id);  // Use sphere light sampling callable

    // Copy light info to GPU
    CUDABuffer<AreaEmitterInfo> d_light_infos;
    d_light_infos.copyToDevice(light_infos);
    m_params.lights = d_light_infos.deviceData();
    m_params.n_lights = static_cast<uint32_t>(light_infos.size());

    CUDA_CHECK(cudaStreamCreate(&m_stream));
    m_scene.copyDataToDevice();
    m_scene.buildAccel(m_ctx, m_stream);
    m_scene.buildSBT();
    m_ppl.create(m_ctx);

    m_params.handle = m_scene.accelHandle();

    // GUI settings
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(pgGetCurrentWindow()->windowPtr(), true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();

    m_scene.launchRay(m_ctx, m_ppl, m_params, m_stream, m_bitmap.width(), m_bitmap.height(), 1);
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    CUDA_SYNC_CHECK();

    m_params.frame++;

    m_bitmap.copyFromDevice();
}

// ------------------------------------------------------------------
void App::draw()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("RTCAMP11");

    const auto& camera = m_scene.camera();
    ImGui::Text("Camera info:");
    ImGui::Text("  Origin: (%.2f, %.2f, %.2f)", camera->origin().x(), camera->origin().y(), camera->origin().z());
    ImGui::Text("  Lookat: (%.2f, %.2f, %.2f)", camera->lookat().x(), camera->lookat().y(), camera->lookat().z());

    ImGui::Text("Frame rate: %.3f ms/frame (%.2f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Subframe index: %d", m_params.frame);

    ImGui::End();
    ImGui::Render();

    m_bitmap.draw(0, 0);

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
    if (key == Key::S)
        m_bitmap.write(pgPathJoin(pgAppDir(), "frame.png"));
}

// ------------------------------------------------------------------
shared_ptr<TriangleMesh> App::buildVoronoiRockMesh()
{
    // Create rock field parameters (2D Voronoi on XZ plane)
    VoronoiRockParams params = createDefaultRockFieldParams();
    params.seed_count = 30;            // Number of rocks to generate
    params.field_size = 20.0f;         // 20x20 field size
    params.y_position = 0.0f;          // Place rocks on ground
    params.min_rock_vertices = 20;     // Minimum vertices per rock
    params.max_rock_vertices = 40;     // Maximum vertices per rock
    params.rock_roughness = 0.3f;      // Surface roughness variation
    params.rock_height_variation = 0.4f; // Y-axis compression variation
    params.random_seed = 54321;

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

    printf("[Voronoi Rock] Generated: %d vertices, %d faces", vertex_count, face_count);

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

    // Create TriangleMesh
    auto mesh = make_shared<TriangleMesh>();
    mesh->addVertices(h_vertices);
    mesh->addNormals(h_normals);
    mesh->addTexcoords(h_texcoords);
    mesh->addFaces(faces);

    mesh->calculateNormalFlat();
    
    // Calculate bounding box to find minimum Y coordinate
    float min_y = h_vertices[0].y();
    float max_y = h_vertices[0].y();
    for (const auto& v : h_vertices) {
        if (v.y() < min_y) min_y = v.y();
        if (v.y() > max_y) max_y = v.y();
    }
    
    printf("[Voronoi Rock] Y-range: %.2f to %.2f (offset needed: %.2f)", min_y, max_y, -min_y);

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
    
    printf("[Terrain] Heightmap texture created: %dx%d (range: %.2f to %.2f)", 
           terrain_data.grid_width, terrain_data.grid_height,
           terrain_data.min_height, terrain_data.max_height);
    
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
pair<shared_ptr<TriangleMesh>, shared_ptr<TriangleMesh>> App::buildTreeMesh(ProceduralTreeData tree, uint32_t& seed, vector<shared_ptr<BitmapTexture>> leaf_textures) {
    // まずCPU版でTree構造を生成（L-systemは再帰的で複雑なのでCPU側で）
    Vec3f origin = Vec3f(0, 0, 0);
    Vec3f dir = Vec3f(0, 1, 0);
    float seg_length = tree.scale / (float)tree.depth;
    float radius = tree.radius;
    const int radial_segments = 12;

    int start_depth = tree.depth;

    // L-system生成（CPU側、既存のbuildTreeStructureをそのままコピー）
    auto buildTreeStructure = [&](auto&& self, Vec3f pos, Vec3f direction, float length, float rad, int depth, uint32_t& seed, TreeSegment* parent_seg = nullptr, float accumulated_twist = 0.0f) -> shared_ptr<TreeBranch> {
        if (depth <= 0) return nullptr;

        auto branch = make_shared<TreeBranch>(tree.depth - depth + 1, length, rad);

        // Stronger taper effect for more pronounced tip tapering
        // Lower values = thinner tips (0.3 means tip is 30% of base radius)
        float taper_factor = 0.4f;

        // ノイズを追加してより有機的な曲線を作成
        const int curve_segments = 4;
        vector<Vec3f> curve_points;
        curve_points.push_back(pos); // 開始点

        Vec3f current_pos = pos;
        Vec3f current_dir = direction;
        float segment_length = length / curve_segments;

        // Depth-based factor for physics forces (stronger effect on younger branches)
        float depth_factor = 1.0f - (float)depth / (float)tree.depth;

        // 各セグメントでノイズと物理力を追加
        for (int i = 1; i <= curve_segments; i++) {
            float t = (float)i / curve_segments;

            // === Apply physics forces ===
            // 1. Gravity (downward force)
            Vec3f gravity_force = Vec3f(0, -tree.gravity * depth_factor, 0) * t;

            // 2. Vertically (upward resistance)
            Vec3f vertical_force = Vec3f(0, tree.vertically * (1.0f - depth_factor), 0) * t;

            // 3. Coverage (horizontal spreading)
            Vec3f horizontal_dir = normalize(Vec3f(current_dir.x(), 0, current_dir.z()));
            if (dot(horizontal_dir, horizontal_dir) < 0.0001f) {
                horizontal_dir = Vec3f(1, 0, 0); // Fallback
            }
            Vec3f coverage_force = horizontal_dir * tree.coverage * depth_factor * t;

            // 4. Random noise for organic variation
            Vec3f noise = Vec3f(
                rnd(seed, -0.3f, 0.3f),
                rnd(seed, -0.2f, 0.2f),
                rnd(seed, -0.3f, 0.3f)
            ) * length * 0.2f;

            // Combine all forces
            Vec3f total_force = gravity_force + vertical_force + coverage_force + noise;

            current_pos += current_dir * segment_length + total_force;
            curve_points.push_back(current_pos);

            // 次のセグメントの方向を更新
            if (i < curve_segments) {
                current_dir = normalize(current_dir + total_force * 0.5f);
            }
        }

        // 最初のセグメント（開始点）を追加
        if (parent_seg) {
            // 親がある場合、親の終端セグメントと完全に一致させる（継ぎ目を滑らかに）
            // 位置、半径、フレーム（normal/binormal）を親から継承
            Vec3f parent_normal = parent_seg->normal;
            Vec3f parent_binormal = parent_seg->binormal;
            Vec3f parent_tangent = parent_seg->direction;

            // Twist rotation around the tangent axis (applied to inherited frame)
            float twist_angle = accumulated_twist;
            Vec3f new_normal = parent_normal * cosf(twist_angle) + parent_binormal * sinf(twist_angle);
            Vec3f new_binormal = cross(parent_tangent, new_normal);
            new_binormal = normalize(new_binormal);
            new_normal = cross(new_binormal, parent_tangent);

            // 最初のセグメントは親と完全に同じ位置・半径・フレームで開始（穴を防ぐ）
            branch->segments.push_back(TreeSegment(pos, parent_tangent, rad, branch->generation, new_normal, new_binormal));
        }
        else {
            // ルートの場合はデフォルト
            branch->segments.push_back(TreeSegment(pos, (curve_points[1] - curve_points[0]), rad, branch->generation));
        }

        // 残りのセグメントを追加（フレーム継承 + twist）
        for (int i = 1; i < curve_segments; i++) {
            Vec3f seg_direction = normalize(curve_points[i + 1] - curve_points[i]);
            float t = (float)(i + 1) / curve_segments;
            float current_radius = rad * (1.0f - t * (1.0f - taper_factor));

            // 前のセグメントのフレームを継承してtwist適用
            TreeSegment& prev_seg = branch->segments.back();

            // Twist angle increases along the segment
            float segment_twist = tree.twist * t * 0.1f; // Small incremental twist per segment
            Vec3f tangent = seg_direction;
            Vec3f rotated_normal = prev_seg.normal * cosf(segment_twist) + prev_seg.binormal * sinf(segment_twist);
            Vec3f rotated_binormal = cross(tangent, rotated_normal);
            rotated_binormal = normalize(rotated_binormal);
            rotated_normal = cross(rotated_binormal, tangent);

            branch->segments.push_back(TreeSegment(curve_points[i + 1], seg_direction, current_radius, branch->generation, rotated_normal, rotated_binormal));
        }

        // 子ブランチを生成
        if (depth > 1) {
            // この枝の最終終端セグメントを親として渡す
            TreeSegment* end_segment = &branch->segments.back();
            Vec3f actual_end_pos = end_segment->position; // 曲線の実際の終端位置

            // 子の開始半径 = 親の終端半径で段差を解消
            float parent_end_radius = end_segment->radius;

            // Accumulate twist for child branches
            float child_twist = accumulated_twist + tree.twist * 0.5f;

            if (tree.trunk_mode && branch->generation == 1) {
                // === Trunk mode: Main trunk continues straight, side branches extend horizontally ===

                // 1. Continue main trunk (1 branch straight up)
                Vec3f trunk_dir = end_segment->direction;
                // Keep trunk mostly vertical with slight variation
                Vec3f trunk_variation = Vec3f(
                    rnd(seed, -0.05f, 0.05f),
                    rnd(seed, 0.0f, 0.1f),
                    rnd(seed, -0.05f, 0.05f)
                );
                trunk_dir = normalize(trunk_dir + trunk_variation);

                float trunk_length = length * 0.95f; // Slightly shorter each level
                float trunk_radius = parent_end_radius * 0.9f; // Thinner as it goes up

                auto trunk_child = self(self, actual_end_pos, trunk_dir, trunk_length, trunk_radius, depth - 1, seed, end_segment, child_twist);
                if (trunk_child) branch->addChild(trunk_child);

                // 2. Add side branches (2-4 branches extending outward)
                int n_side_branches = rndInt(seed, 2, 4);
                float side_branch_length = length * (0.4f + rnd(seed) * 0.2f); // Shorter than trunk
                float side_branch_radius = parent_end_radius * (0.3f + rnd(seed) * 0.2f);

                for (int i = 0; i < n_side_branches; i++) {
                    // Generate side branches extending outward at angle
                    float angle = rnd(seed) * 2.0f * math::pi; // Random angle around trunk
                    float upward_angle = rnd(seed, -0.2f, 0.4f); // Slightly upward or horizontal

                    Vec3f radial_dir = Vec3f(cosf(angle), 0, sinf(angle));
                    Vec3f side_dir = normalize(Vec3f(
                        radial_dir.x() * (0.7f + tree.coverage * 0.3f),
                        upward_angle,
                        radial_dir.z() * (0.7f + tree.coverage * 0.3f)
                    ));

                    auto side_child = self(self, actual_end_pos, side_dir, side_branch_length, side_branch_radius, depth - 2, seed, end_segment, child_twist);
                    if (side_child) branch->addChild(side_child);
                }
            }
            else {
                // === Normal branching mode: All branches diverge ===
                int n_branches = rndInt(seed, 3, 6);
                float new_length = length * (0.7f + rnd(seed) * 0.3f);

                for (int i = 0; i < n_branches; i++) {
                    // ランダムな方向（曲線終端の方向を基準に）
                    Vec3f base_dir = end_segment->direction;

                    // Apply coverage to child branch directions (spread horizontally)
                    Vec3f random_offset = Vec3f(
                        rnd(seed, -0.6f, 0.6f),
                        rnd(seed, -0.4f, 0.4f),
                        rnd(seed, -0.6f, 0.6f)
                    );

                    // Enhance horizontal spread with coverage parameter
                    random_offset.x() *= (1.0f + tree.coverage * 0.5f);
                    random_offset.z() *= (1.0f + tree.coverage * 0.5f);

                    Vec3f new_dir = normalize(base_dir + random_offset);

                    // 親の終端半径から開始して滑らかに接続
                    auto child = self(self, actual_end_pos, new_dir, new_length, parent_end_radius, depth - 1, seed, end_segment, child_twist);
                    if (child) branch->addChild(child);
                }
            }
        }

        return branch;
        };

    // Generate tree structure (root has no parent, no initial twist)
    shared_ptr<TreeBranch> root = buildTreeStructure(buildTreeStructure, origin, dir, seg_length, radius, tree.depth, seed, nullptr, 0.0f);

    // Convert to CUDA format
    std::vector<CudaTreeBranch> cuda_branches;
    std::vector<CudaTreeSegment> cuda_segments;
    convertTreeToCudaFormat(root, cuda_branches, cuda_segments);

    // GPUメモリに転送
    CudaTreeBranch* d_branches;
    CudaTreeSegment* d_segments;
    cudaMalloc(&d_branches, cuda_branches.size() * sizeof(CudaTreeBranch));
    cudaMalloc(&d_segments, cuda_segments.size() * sizeof(CudaTreeSegment));
    cudaMemcpy(d_branches, cuda_branches.data(), cuda_branches.size() * sizeof(CudaTreeBranch), cudaMemcpyHostToDevice);
    cudaMemcpy(d_segments, cuda_segments.data(), cuda_segments.size() * sizeof(CudaTreeSegment), cudaMemcpyHostToDevice);

    // CUDA kernelでメッシュ化
    Vec3f* d_vertices, * d_normals;
    Vec2f* d_texcoords;
    Vec3i* d_face_indices, * d_normal_indices, * d_texcoord_indices;
    int vertex_count, face_count;
    int branch_count = (int)cuda_branches.size();
    int segment_count = (int)cuda_segments.size();

    buildTreeMeshCUDA(
        d_branches, d_segments,
        branch_count, segment_count,
        radial_segments,
        &d_vertices, &d_normals, &d_texcoords,
        &d_face_indices, &d_normal_indices, &d_texcoord_indices,
        &vertex_count, &face_count
    );

    // GPUからCPUにデータを転送
    std::vector<Vec3f> vertices(vertex_count);
    std::vector<Vec3f> normals(vertex_count);
    std::vector<Vec2f> texcoords(vertex_count);
    std::vector<Face> faces(face_count);

    cudaMemcpy(vertices.data(), d_vertices, vertex_count * sizeof(Vec3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(normals.data(), d_normals, vertex_count * sizeof(Vec3f), cudaMemcpyDeviceToHost);
    cudaMemcpy(texcoords.data(), d_texcoords, vertex_count * sizeof(Vec2f), cudaMemcpyDeviceToHost);

    // Faceデータの変換
    std::vector<Vec3i> temp_face_indices(face_count);
    cudaMemcpy(temp_face_indices.data(), d_face_indices, face_count * sizeof(Vec3i), cudaMemcpyDeviceToHost);

    for (int i = 0; i < face_count; i++) {
        faces[i] = { temp_face_indices[i], temp_face_indices[i], temp_face_indices[i] };
    }

    // メッシュ作成
    auto tree_mesh = make_shared<TriangleMesh>();
    tree_mesh->addVertices(vertices);
    tree_mesh->addNormals(normals);
    tree_mesh->addTexcoords(texcoords);
    tree_mesh->addFaces(faces);

    // 葉っぱメッシュを別途作成（CUDA版）
    shared_ptr<TriangleMesh> leaf_mesh = nullptr;
    
    if (tree.has_leaves && !leaf_textures.empty()) {
        // CUDA葉っぱ生成パラメータを設定
        LeafGenerationParams leaf_params;
        leaf_params.leaf_density = tree.leaf_density;
        leaf_params.leaf_start_gen = tree.leaf_start_gen;
        leaf_params.leaf_size = tree.leaf_size;
        leaf_params.seed = seed;
        leaf_params.num_leaf_textures = static_cast<int>(leaf_textures.size());

        // デバイスポインタ（出力）
        Vec3f* d_leaf_vertices = nullptr;
        Vec3f* d_leaf_normals = nullptr;
        Vec2f* d_leaf_texcoords = nullptr;
        Vec3i* d_leaf_face_indices = nullptr;
        Vec3i* d_leaf_normal_indices = nullptr;
        Vec3i* d_leaf_texcoord_indices = nullptr;
        uint32_t* d_leaf_sbt_indices = nullptr;
        int leaf_vertex_count = 0;
        int leaf_face_count = 0;

        // CUDAで葉っぱメッシュ生成
        buildLeafMeshCUDA(
            d_branches, d_segments,
            branch_count, segment_count,
            leaf_params,
            &d_leaf_vertices, &d_leaf_normals, &d_leaf_texcoords,
            &d_leaf_face_indices, &d_leaf_normal_indices, &d_leaf_texcoord_indices,
            &d_leaf_sbt_indices,
            &leaf_vertex_count, &leaf_face_count
        );
        
        // Check for CUDA errors after leaf generation
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[CUDA Error] buildLeafMeshCUDA failed: %s", cudaGetErrorString(err));
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // デバイスからホストへコピー
        if (leaf_vertex_count > 0 && leaf_face_count > 0) {
            vector<Vec3f> leaf_vertices(leaf_vertex_count);
            vector<Vec3f> leaf_normals(leaf_vertex_count);
            vector<Vec2f> leaf_texcoords(leaf_vertex_count);
            vector<Vec3i> temp_leaf_face_indices(leaf_face_count);
            vector<Face> leaf_faces(leaf_face_count);

            cudaMemcpy(leaf_vertices.data(), d_leaf_vertices, 
                       leaf_vertex_count * sizeof(Vec3f), cudaMemcpyDeviceToHost);
            cudaMemcpy(leaf_normals.data(), d_leaf_normals, 
                       leaf_vertex_count * sizeof(Vec3f), cudaMemcpyDeviceToHost);
            cudaMemcpy(leaf_texcoords.data(), d_leaf_texcoords, 
                       leaf_vertex_count * sizeof(Vec2f), cudaMemcpyDeviceToHost);
            cudaMemcpy(temp_leaf_face_indices.data(), d_leaf_face_indices, 
                       leaf_face_count * sizeof(Vec3i), cudaMemcpyDeviceToHost);

            // Vec3iからFaceに変換
            for (int i = 0; i < leaf_face_count; i++) {
                leaf_faces[i] = { temp_leaf_face_indices[i], temp_leaf_face_indices[i], temp_leaf_face_indices[i] };
            }

            // 葉っぱメッシュを作成
            leaf_mesh = make_shared<TriangleMesh>();
            leaf_mesh->addVertices(leaf_vertices);
            leaf_mesh->addNormals(leaf_normals);
            leaf_mesh->addTexcoords(leaf_texcoords);
            leaf_mesh->addFaces(leaf_faces);

            // SBTインデックスをランダムに設定（マルチテクスチャ対応）
            if (!leaf_textures.empty()) {
                vector<uint32_t> sbt_indices(leaf_face_count);
                uint32_t sbt_seed = seed + 999999;  // 別のseed値
                for (int i = 0; i < leaf_face_count; i++) {
                    sbt_indices[i] = rndInt(sbt_seed, 0, (int)leaf_textures.size() - 1);
                }
                leaf_mesh->setSbtIndices(sbt_indices);
            }

            printf("CUDA Leaf Generation: %d vertices, %d faces, %d textures", 
                   leaf_vertex_count, leaf_face_count, (int)leaf_textures.size());
        }

        // CUDAメモリ解放
        if (d_leaf_vertices) cudaFree(d_leaf_vertices);
        if (d_leaf_normals) cudaFree(d_leaf_normals);
        if (d_leaf_texcoords) cudaFree(d_leaf_texcoords);
        if (d_leaf_face_indices) cudaFree(d_leaf_face_indices);
        if (d_leaf_normal_indices) cudaFree(d_leaf_normal_indices);
        if (d_leaf_texcoord_indices) cudaFree(d_leaf_texcoord_indices);
        if (d_leaf_sbt_indices) cudaFree(d_leaf_sbt_indices);
    }

    // GPUメモリ解放
    cudaFree(d_branches);
    cudaFree(d_segments);
    cudaFree(d_vertices);
    cudaFree(d_normals);
    cudaFree(d_texcoords);
    cudaFree(d_face_indices);
    cudaFree(d_normal_indices);
    cudaFree(d_texcoord_indices);

    return {tree_mesh, leaf_mesh};
}

// ------------------------------------------------------------------
// New Tree API version for testing
// ------------------------------------------------------------------
pair<shared_ptr<TriangleMesh>, shared_ptr<TriangleMesh>> App::buildTreeMeshWithAPI(uint32_t& seed) {
    // Create Tree object with parameters
    Tree tree(seed);
    TreeParam& params = tree.getParams();
    
    // Customize some parameters (use defaults for most)
    params.g_scale = 30.0f;
    params.g_scale_v = 5.0f;
    params.levels = 3;
    params.shape = 8;
    params.branches[0] = 1;  // Single trunk
    params.branches[1] = 10;
    params.branches[2] = 10;
    params.branches[3] = 10;
    
    // Generate tree structure (CPU side)
    tree.createBranches();
    
    const auto& branch_curves = tree.getBranchCurves();
    const auto& leaves = tree.getLeaves();
    
    // Count total splines across all levels
    int total_splines = 0;
    for (const auto& curve : branch_curves) {
        total_splines += curve->splines.size();
    }
    
    pgLog("New Tree API generated: " + to_string(branch_curves.size()) + " levels, " + 
          to_string(total_splines) + " splines, " + to_string(leaves.size()) + " leaves");
    
    // Convert Bezier curves to mesh
    auto tree_mesh = make_shared<TriangleMesh>();
    shared_ptr<TriangleMesh> leaf_mesh = nullptr;
    
    vector<Vec3f> vertices;
    vector<Vec3f> normals;
    vector<Vec2f> texcoords;
    vector<Face> faces;
    
    int radial_segments = 8;  // Number of vertices around the branch
    
    // Process each level of branches
    for (const auto& branch_curve : branch_curves) {
        // Process each spline in this level
        for (const auto& spline : branch_curve->splines) {
            if (spline->bezier_points.empty()) continue;
            
            // Debug: Check bezier points
            pgLog(std::format("Spline has {} bezier points:", spline->bezier_points.size()));
            for (size_t i = 0; i < std::min(size_t(3), spline->bezier_points.size()); i++) {
                const auto& bp = spline->bezier_points[i];
                pgLog(std::format("  Point[{}]: co=({:.3f}, {:.3f}, {:.3f}), radius={:.3f}", 
                    i, bp.co.x(), bp.co.y(), bp.co.z(), bp.radius));
            }
            
            int num_points = spline->bezier_points.size();
            int samples_per_segment = 4;  // Samples between control points
            
            // Sample the bezier curve
            for (int seg = 0; seg < num_points - 1; seg++) {
                for (int sample = 0; sample < samples_per_segment; sample++) {
                    float t_local = float(sample) / samples_per_segment;
                    float t_global = (seg + t_local) / (num_points - 1);
                    
                    Vec3f pos = spline->evaluate(t_global);
                    float radius = spline->evaluateRadius(t_global);

                    pgLog(std::format("Spline Segment: {}, Sample: {}, t_global: {:.3f}, Pos: ({:.2f}, {:.2f}, {:.2f}), Radius: {:.3f}",
                                     seg, sample, t_global, pos.x(), pos.y(), pos.z(), radius));
                    
                    // Calculate tangent for orientation
                    float dt = 0.01f;
                    Vec3f pos_next = spline->evaluate(std::min(t_global + dt, 1.0f));
                    Vec3f tangent = normalize(pos_next - pos);
                    
                    // Create perpendicular frame
                    Vec3f up = fabs(tangent.y()) < 0.99f ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0);
                    Vec3f right = normalize(cross(tangent, up));
                    Vec3f forward = normalize(cross(right, tangent));
                    
                    int base_vertex = (int)vertices.size();
                    
                    // Create ring of vertices around this point
                    for (int i = 0; i < radial_segments; i++) {
                    float angle = 2.0f * math::pi * i / radial_segments;
                    float x = cos(angle);
                    float z = sin(angle);
                    
                    Vec3f offset = (right * x + forward * z) * radius;
                    Vec3f vertex_pos = pos + offset;
                    Vec3f normal = normalize(offset);
                    
                    vertices.push_back(vertex_pos);
                    normals.push_back(normal);
                    texcoords.push_back(Vec2f(float(i) / radial_segments, t_global));
                }
                //pgLog(std::format("Vertices size: {}, Normals size: {}, Texcoords size: {}", vertices.size(), normals.size(), texcoords.size()));
                
                // Create faces connecting to previous ring
                if (base_vertex >= radial_segments) {
                    int prev_base = base_vertex - radial_segments;
                    
                    for (int i = 0; i < radial_segments; i++) {
                        int next_i = (i + 1) % radial_segments;
                        
                        int v0 = prev_base + i;
                        int v1 = prev_base + next_i;
                        int v2 = base_vertex + next_i;
                        int v3 = base_vertex + i;
                        
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
                        //pgLog(std::format("v0: {}, v1: {}, v2: {}, v3: {}", v0, v1, v2, v2));
                    }
                }
            }
        }
    }
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

            // Use Leaf class API: position(), direction(), right()
            Vec3f pos = leaf.position();
            Vec3f normal = leaf.direction();

            // Leaf size is controlled by params; per-leaf scale removed (was in LeafData)
            Vec3f up = normal * leaf_size;

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

            Vec3f right = right_dir * leaf_size * leaf_scale_x;

            // Four corners of leaf quad (centered at pos)
            leaf_vertices.push_back(pos - right - up * 0.5f);
            leaf_vertices.push_back(pos + right - up * 0.5f);
            leaf_vertices.push_back(pos + right + up * 0.5f);
            leaf_vertices.push_back(pos - right + up * 0.5f);

            for (int i = 0; i < 4; i++) {
                leaf_normals.push_back(normal);
            }

            leaf_texcoords.push_back(Vec2f(0, 0));
            leaf_texcoords.push_back(Vec2f(1, 0));
            leaf_texcoords.push_back(Vec2f(1, 1));
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
            pgLog("New Tree API: Created leaf mesh with " + to_string(leaf_vertices.size()) + " vertices");
        }
    }
    
    if (!vertices.empty()) {
        tree_mesh->addVertices(vertices);
        tree_mesh->addNormals(normals);
        tree_mesh->addTexcoords(texcoords);
        tree_mesh->addFaces(faces);
        pgLog("New Tree API: Created tree mesh with " + to_string(vertices.size()) + " vertices, " + to_string(faces.size()) + " faces");
    }
    
    return {tree_mesh, leaf_mesh};
}



