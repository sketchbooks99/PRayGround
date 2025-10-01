#include "app.h"

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
shared_ptr<Curves> App::buildTreeCurves(ProceduralTreeData tree)
{
    uint32_t seed = tea<4>(0, 0);

    Vec3f origin = Vec3f(0, 0, 0);
    Vec3f dir = Vec3f(0, 1, 0);

    float length = tree.scale / (float)tree.depth;

    float radius = length * 0.1f;

    auto curve = make_shared<Curves>(Curves::Type::CubicBSpline);

    // Build tree by L-system (SIMPLIFIED VERSION)
    Matrix4f mat = Matrix4f::identity();

    auto buildBranch = [&](auto&& self, int depth, Vec3f pos, Vec3f dir, float length, float radius, uint32_t& seed, Matrix4f mat, bool is_continuous = false, int parent_end_idx = -1) -> int {
        if (depth == 0) return -1;

        int n_vertices = 4; // Simple: 4 vertices per segment (no phantoms)
        int n_segments = n_vertices - (int)Curves::getNumVertexPerSegment(curve->curveType()) + 1;
        
        int segment_start_idx;
        
        if (parent_end_idx == -1) {
            // ROOT BRANCH: Add phantom start + real vertices + phantom end
            Vec3f phantom_start = pos - dir * (length * 0.2f);
            curve->addVertex(phantom_start);
            curve->addWidth(radius);
            segment_start_idx = curve->vertices().size() - 1;
            
            // Add real vertices
            for (int i = 0; i < n_vertices; i++) {
                float t = (float)i / (float)(n_vertices - 1);
                Vec3f vertex_pos = pos + dir * (length * t);
                curve->addVertex(vertex_pos);
                curve->addWidth(radius * (1.0f - t * 0.4f));
            }
            
            // Add phantom end
            Vec3f phantom_end = pos + dir * length + dir * (length * 0.2f);
            curve->addVertex(phantom_end);
            curve->addWidth(radius * 0.6f);
            
            printf("ROOT: phantom_start=%d, vertices=[%d-%d], phantom_end=%d\n", 
                   segment_start_idx, segment_start_idx + 1, segment_start_idx + n_vertices, (int)curve->vertices().size() - 1);
                   
        } else {
            // CHILD BRANCH: No phantoms, simple connection
            if (is_continuous) {
                // Continuous: start from parent's end vertex
                segment_start_idx = parent_end_idx;
                // Add only NEW vertices (skip first since shared with parent)
                for (int i = 1; i < n_vertices; i++) {
                    float t = (float)i / (float)(n_vertices - 1);
                    Vec3f vertex_pos = pos + dir * (length * t);
                    curve->addVertex(vertex_pos);
                    curve->addWidth(radius * (1.0f - t * 0.4f));
                }
                printf("CONTINUOUS: start=%d, added vertices=[%d-%d]\n", 
                       segment_start_idx, parent_end_idx + 1, (int)curve->vertices().size() - 1);
            } else {
                // Independent: start fresh
                segment_start_idx = curve->vertices().size();
                // Add ALL vertices
                for (int i = 0; i < n_vertices; i++) {
                    float t = (float)i / (float)(n_vertices - 1);
                    Vec3f vertex_pos = pos + dir * (length * t);
                    curve->addVertex(vertex_pos);
                    curve->addWidth(radius * (1.0f - t * 0.4f));
                }
                printf("INDEPENDENT: start=%d, vertices=[%d-%d]\n", 
                       segment_start_idx, segment_start_idx, (int)curve->vertices().size() - 1);
            }
        }
        
        // Add indices (simple sequential, NO phantoms)
        if (depth == 1) {
            for (int i = 0; i < n_segments + 2; i++) 
               curve->addIndex(segment_start_idx + i);
        } else {
            int add_num_index = parent_end_idx == -1 ? n_vertices + 2 : n_vertices;
            for (int i = 0; i < add_num_index; i++) 
               curve->addIndex(segment_start_idx + i);
        }
        printf("Indices: %d to %d\n", segment_start_idx, segment_start_idx + n_segments - 1);
        
        // End vertex and position
        int my_end_vertex_idx = curve->vertices().size() - 1;
        Vec3f branch_end_pos = pos + dir * length;
        
        // Generate child branches
        int depth_ = depth - 1;
        length *= (0.7f + rnd(seed) * 0.3f);
        radius *= 0.7f;
        
        int n_branches = rndInt(seed, 2, 4);
        for (int i = 0; i < n_branches; i++) {
            Matrix4f rot_x = Matrix4f::rotate(math::pi * rnd(seed, -0.15f, 0.15f), Vec3f(1, 0, 0));
            Matrix4f rot_z = Matrix4f::rotate(math::pi * rnd(seed, -0.15f, 0.15f), Vec3f(0, 0, 1));
            Matrix4f rot_y = Matrix4f::rotate(math::pi * rnd(seed, -0.1f, 0.1f), Vec3f(0, 1, 0));
            Matrix4f new_mat = rot_x * rot_y * rot_z * mat;
            Vec3f new_dir = normalize(new_mat.vectorMul(dir) + Vec3f(0, tree.gravity, 0) + Vec3f((rnd(seed) - 0.5f) * tree.horizontal_strength, 0, (rnd(seed) - 0.5f) * tree.horizontal_strength));
            
            if (i == 0) {
                // First child: CONTINUOUS (shares end vertex)
                self(self, depth_, branch_end_pos, new_dir, length, radius, seed, new_mat, true, my_end_vertex_idx + 1);
            } else {
                // Other children: INDEPENDENT (new vertices)
                self(self, depth_, branch_end_pos, new_dir, length, radius, seed, new_mat, false, -1);
            }
        }
        
        return my_end_vertex_idx;
    };
    // Main trunk (root branch)
    buildBranch(buildBranch, tree.depth, origin, dir, length, radius, seed, mat, false, -1);

    for (auto& v : curve->indices())
        cout << v << ", ";
    cout << endl;
    return curve;
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

    // Miss program & environemnt map
    array<ProgramGroup, NRay> miss_prgs;
    miss_prgs[0] = m_ppl.createMissProgram(m_ctx, module, "__miss__envmap");
    miss_prgs[1] = m_ppl.createMissProgram(m_ctx, module, "__miss__shadow");
    m_scene.bindMissPrograms(miss_prgs);
    auto env_texture = make_shared<ConstantTexture>(Vec3f(0.5f), constant_id);
    m_scene.setEnvmap(env_texture);

    // Hitgroup programs
    array<ProgramGroup, NRay> mesh_prgs;
    mesh_prgs[0] = m_ppl.createHitgroupProgram(m_ctx, module, "__closesthit__mesh");
    mesh_prgs[1] = m_ppl.createHitgroupProgram(m_ctx, module, "__closesthit__shadow");
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

    Callable area_emitter_prg = m_ppl.createCallablesProgram(m_ctx, module, "__direct_callable__area_emitter", "");

    // Textures
    auto procedural_wooden_texture = make_shared<ProceduralWoodenTexture>(
        Vec3f(0.76f, 0.60f, 0.42f), Vec3f(0.4f, 0.2f, 0.1f),
        0.1f, 20.0f, 0.2f,
        procedural_wooden_id
    );
    auto checker_texture = make_shared<CheckerTexture>(
        Vec3f(0.2f), Vec3f(0.8f), 10.0f, checker_id
    );

    // Surfaces
    auto floor_diffuse = make_shared<Diffuse>(diffuse_id, checker_texture);
    auto tree_diffuse = make_shared<Diffuse>(diffuse_id, procedural_wooden_texture);

    m_scene.addObject("floor", make_shared<Plane>(Vec2f(-100), Vec2f(100)), floor_diffuse, plane_prgs, Matrix4f::identity());
    //m_scene.addObject("sphere", make_shared<Sphere>(Vec3f(0.0f, 10.0f, 0.0f), 10.0f), floor_diffuse, sphere_prgs, Matrix4f::identity());
    ProceduralTreeData tree = {
        .gravity = 0.0f,
        .horizontal_strength = 0.5f,
        .scale = 50.0f,
        .depth = 5
    };
    auto tree_curve = buildTreeCurves(tree);
    m_scene.addObject("tree", tree_curve, tree_diffuse, curve_prgs, Matrix4f::identity());

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

}



