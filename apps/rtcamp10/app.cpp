#include "app.h"

#include <chrono>

void App::initResultBufferOnDevice() 
{
    params.frame = 0u;

    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();

    params.result_buffer = (Vec4u*)result_bmp.deviceData();
    params.accum_buffer = (Vec4f*)accum_bmp.deviceData();
}

void App::handleCameraUpdate() 
{
    if (!is_camera_updated) return;
    is_camera_updated = false;

    scene.updateSBT(+(SBTRecordType::Raygen));

    initResultBufferOnDevice();
}

// ------------------------------------------------------------------
void App::setup()
{
    using namespace std::chrono;
    system_clock::time_point start = system_clock::now();

    // Initialize CUDA 
    stream = 0;
    CUDA_CHECK(cudaFree(0));

    // Initialize OptiX
    OPTIX_CHECK(optixInit());
    ctx.disableValidation();
    ctx.create();

    // Initialize pipeline
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(10);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);

    // Create module
    Module module = pipeline.createModuleFromOptixIr(ctx, "ptx/Release/rtcamp10_generated_kernels.cu.optixir");

    // Initialize bitmap 
    const int32_t width = pgGetWidth();
    const int32_t height = pgGetHeight();
    result_bmp.allocate(PixelFormat::RGBA, width, height);
    accum_bmp.allocate(PixelFormat::RGBA, width, height);
    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();

    // Launch parameter initialization
    params.width = width;
    params.height = height;
#if SUBMISSION
    params.samples_per_launch = 32;
#else
    params.samples_per_launch = 8;
#endif
    params.frame = 0u;
    params.max_depth = 10;
    params.result_buffer = (Vec4u*)result_bmp.deviceData();
    params.accum_buffer = (Vec4f*)accum_bmp.deviceData();

    // Setup scene
    Scene<Camera, NRay>::AccelSettings accel_settings;
    accel_settings.allow_accel_compaction = true;
    accel_settings.allow_accel_update = true;
    scene.setup(accel_settings);

    // Setup camera
    shared_ptr<Camera> camera = make_shared<Camera>();
    camera->setOrigin(0, -50, 50);
    camera->setLookat(0, -80, 0);
    camera->setUp(0, 1, 0);
    camera->setFov(40);
    camera->setAspect((float)width / height);
#if !SUBMISSION
    camera->enableTracking(pgGetCurrentWindow());
#endif
    scene.setCamera(camera);

    // Raygen program
    ProgramGroup raygen = pipeline.createRaygenProgram(ctx, module, "__raygen__pinhole");
    scene.bindRaygenProgram(raygen);

    struct Callable {
        Callable(const pair<ProgramGroup, uint32_t>& callable)
            : program(callable.first), ID(callable.second) {};
        ProgramGroup program;
        uint32_t ID;
    };

    auto setupCallable = [&](const string& dc, const string& cc) -> uint32_t {
        Callable callable = pipeline.createCallablesProgram(ctx, module, dc, cc);
        scene.bindCallablesProgram(callable.program);
        return callable.ID;
    };

    // Texture programs
    auto bitmap_id = setupCallable("__direct_callable__bitmap", "");
    auto checker_id = setupCallable("__direct_callable__checker", "");
    auto constant_id = setupCallable("__direct_callable__constant", "");

    // Miss program
    ProgramGroup miss = pipeline.createMissProgram(ctx, module, "__miss__envmap");
    ProgramGroup miss_shadow = pipeline.createMissProgram(ctx, module, "__miss__shadow");
    scene.bindMissPrograms({ miss, miss_shadow });
    auto envmap_texture = make_shared<FloatBitmapTexture>("resources/drackenstein_quarry_4k.exr", bitmap_id);
    //auto envmap_texture = make_shared<ConstantTexture>(Vec3f(0.0f), constant_id);
    scene.setEnvmap(envmap_texture);
    // TODO: Create Distribution2D for envmap importance sampling

    // Hitgroup programs
    // Mesh
    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__mesh");
    ProgramGroup mesh_shadow_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__shadow");
    // Sphere
    ProgramGroup sphere_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__custom", "__intersection__sphere");
    ProgramGroup sphere_shadow_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__shadow", "__intersection__sphere");
    // Plane
    ProgramGroup plane_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__custom", "__intersection__plane");
    ProgramGroup plane_shadow_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__shadow", "__intersection__plane");

    // Surface programs
    SurfaceCallableID diffuse_id = { 
        setupCallable("__direct_callable__sample_diffuse", ""), 
        setupCallable("__direct_callable__bsdf_diffuse", ""),
        setupCallable("__direct_callable__pdf_diffuse", "")
    };
    SurfaceCallableID conductor_id = {
        setupCallable("__direct_callable__sample_conductor", ""),
        setupCallable("__direct_callable__bsdf_conductor", ""),
        setupCallable("__direct_callable__pdf_conductor", "")
    };
    SurfaceCallableID dielectric_id = {
        setupCallable("__direct_callable__sample_dielectric", ""),
        setupCallable("__direct_callable__bsdf_dielectric", ""),
        setupCallable("__direct_callable__pdf_dielectric", "")
    };
    SurfaceCallableID disney_id = {
        setupCallable("__direct_callable__sample_disney", ""),
        setupCallable("__direct_callable__bsdf_disney", ""),
        setupCallable("__direct_callable__pdf_disney", "")
    };
    SurfaceCallableID layered_id = {
        setupCallable("__direct_callable__sample_layered", ""),
        setupCallable("__direct_callable__bsdf_layered", ""),
        setupCallable("__direct_callable__pdf_layered", "")
    };
    SurfaceCallableID area_emitter_id = { 
        setupCallable("__direct_callable__area_emitter", "")
    };

    // Create surfaces
    auto floor_mat = make_shared<Diffuse>(diffuse_id, make_shared<CheckerTexture>(Vec3f(0.8f), Vec3f(0.2f), 10, checker_id));
    auto sphere_mat = make_shared<Dielectric>(dielectric_id,
        /* texture = */ make_shared<ConstantTexture>(Vec3f(1.0f, 1.0f, 1.0f), constant_id),
        /* ior = */ 1.33f,
        /* absorb_coeff = */ 0.0f,
        /* sellmeier = */ Sellmeier::None, 
        Thinfilm(
            /* ior = */ Vec3f(1.33f),
            /* thickness = */ make_shared<ConstantTexture>(Vec3f(1.0f), constant_id),
            /* thickness_scale = */ 550.0f,
            /* tf_ior = */ 1.7f,
            /* extinction = */ Vec3f(1.0f)
        ));
    auto bunny_mat = make_shared<Conductor>(conductor_id, 
        /* texture = */ make_shared<ConstantTexture>(Vec3f(0.8f, 0.8f, 0.3f), constant_id), 
        /* twosided = */ true, 
        Thinfilm(
            /* ior = */ Vec3f(1.9f),
            /* thickness = */ make_shared<ConstantTexture>(Vec3f(1.0f), constant_id),
            /* thickness_scale = */ 550.0f,
            /* tf_ior = */ 1.33f,
            /* extinction = */ Vec3f(1.5f)
        ));

    auto katsuo_base = make_shared<Diffuse>(diffuse_id, make_shared<BitmapTexture>("resources/Katsuo Color.png", bitmap_id));
    auto katsuo_oil = make_shared<Dielectric>(dielectric_id,
        make_shared<ConstantTexture>(Vec3f(0.9f, 0.9f, 0.8f), constant_id),
        1.5f, 0.0f, Sellmeier::None,
        Thinfilm(Vec3f(1.0f), make_shared<BitmapTexture>("resources/Katsuo thinfilm thickness.png", bitmap_id), 1000.0f, 2.45f, Vec3f(3.0f, 6.0f, 2.0f)));
    std::vector <std::shared_ptr<Material>> layers = { katsuo_oil, katsuo_base };
    auto katsuo_mat = make_shared<Layered>(layered_id, layers);
    katsuo_mat->setBumpmap(make_shared<BitmapTexture>("resources/Katsuo Normal.png", bitmap_id));

    auto table_mat = make_shared<Disney>(disney_id, make_shared<BitmapTexture>("resources/Table Color.png", bitmap_id));
    table_mat->setSpecular(0.5f);
    table_mat->setRoughness(0.5f);
    table_mat->setMetallic(0.0f);
    table_mat->setSheenTint(0.5f);

    auto sashimi_plate_mat = make_shared<Disney>(disney_id, make_shared<ConstantTexture>(Vec3f(1.0f), constant_id));
    sashimi_plate_mat->setMetallic(0.1f);
    sashimi_plate_mat->setSpecular(0.9f);
    sashimi_plate_mat->setRoughness(0.4f);

    auto light1 = make_shared<AreaEmitter>(area_emitter_id, make_shared<ConstantTexture>(Vec3f(1.0f), constant_id), 100.0f);

    // Setup geometries in the scene
    // Floor geometry
    //shared_ptr<Plane> plane = make_shared<Plane>(Vec2f(-200), Vec2f(200));
    //scene.addObject("floor", plane, floor_mat, { plane_prg, plane_shadow_prg }, Matrix4f::translate(0, -100, 0));

    // Sphere geometry
    shared_ptr<Sphere> sphere = make_shared<Sphere>(10);
    scene.addObject("sphere", sphere, sphere_mat, { sphere_prg, sphere_shadow_prg }, Matrix4f::translate(-10, -80, 0));

    shared_ptr<TriangleMesh> bunny = make_shared<TriangleMesh>("resources/uv_bunny.obj");
    bunny->calculateNormalSmooth();
    scene.addObject("bunny", bunny, bunny_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(10, -80, 0) * Matrix4f::scale(100));

    shared_ptr<TriangleMesh> katsuo = make_shared<TriangleMesh>("resources/Katsuo.obj");
    scene.addObject("katsuo", katsuo, katsuo_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(30, -80, 0) * Matrix4f::scale(10));

    shared_ptr<TriangleMesh> table = make_shared<TriangleMesh>("resources/Table.obj");
    scene.addObject("table", table, table_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(0, -90, 0)* Matrix4f::scale(10));

    shared_ptr<TriangleMesh> sashimi_plate = make_shared<TriangleMesh>("resources/Sashimi plate.obj");
    scene.addObject("sashimi plate", sashimi_plate, sashimi_plate_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(0, -90, 0)* Matrix4f::scale(10));

    shared_ptr<Sphere> light1_geom = make_shared<Sphere>(5);
    scene.addLight("light1", light1_geom, light1, { sphere_prg, sphere_shadow_prg }, Matrix4f::translate(50, 20, 30));
    scene.addLight("light2", light1_geom, light1, { sphere_prg, sphere_shadow_prg }, Matrix4f::translate(-50, 20, 30));
    scene.addLight("light3", light1_geom, light1, { sphere_prg, sphere_shadow_prg }, Matrix4f::translate(-50, 20, -30));
    scene.addLight("light4", light1_geom, light1, { sphere_prg, sphere_shadow_prg }, Matrix4f::translate(50, 20, -30));

    // Setup context
    CUDA_CHECK(cudaStreamCreate(&stream));
    scene.copyDataToDevice();
    scene.buildAccel(ctx, stream);
    scene.buildSBT();
    pipeline.create(ctx);

    params.handle = scene.accelHandle();

#if SUBMISSION
    double elapsed_seconds = 0.0;
    int32_t frame = 0;

    while (elapsed_seconds < 256.0) {
        elapsed_seconds = (double)duration_cast<milliseconds>(system_clock::now() - start).count() / 1000;
        if (elapsed_seconds > 256.0) break;

        scene.launchRay(ctx, pipeline, params, stream, result_bmp.width(), result_bmp.height(), 1);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_SYNC_CHECK();

        result_bmp.copyFromDevice();

        string filename = std::format("image_{:03d}.png", frame);
        filesystem::path filepath = pgPathJoin(pgGetExecutableDir(), filename);
        result_bmp.write(filepath);

        const float x = sinf(elapsed_seconds) * 50;
        const float z = cosf(elapsed_seconds) * 50;
        camera->setOrigin(x, -50, z);
        camera->setLookat(0, -80, 0);
        is_camera_updated = true;
        handleCameraUpdate();

        std::cout << elapsed_seconds << "s" << std::flush;
        
        frame++;
    }
    cout << endl;
#else
    // GUI initialization
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    const char* glsl_version = "#version 330";
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(pgGetCurrentWindow()->windowPtr(), true);
    ImGui_ImplOpenGL3_Init(glsl_version);
#endif
}


// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();

    scene.launchRay(ctx, pipeline, params, stream, result_bmp.width(), result_bmp.height(), 1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    result_bmp.copyFromDevice();

    params.frame++;
}

// ------------------------------------------------------------------
void App::draw()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("RTCAMP 10 editor");

    // Update katsuo object
    auto katsuo = scene.getObject("katsuo");
    auto katsuo_mat = dynamic_pointer_cast<Layered>(katsuo->materials[0]);
    auto katsuo_top_layer = dynamic_pointer_cast<Dielectric>(katsuo_mat->layerAt(0));
    Thinfilm thinfilm = katsuo_top_layer->thinfilm();
    Vec3f ior = thinfilm.ior();
    Vec3f extinction = thinfilm.extinction();
    float thickness_scale = thinfilm.thicknessScale();
    float tf_ior = thinfilm.tfIor();

    if (ImGui::TreeNode("Katsuo")) {
        if (ImGui::SliderFloat3("IOR", &ior[0], 1.0f, 20.0f)) {
            thinfilm.setIor(ior);
            katsuo_top_layer->setThinfilm(thinfilm);
            is_camera_updated = true;
        }
        if (ImGui::SliderFloat3("Extinction", &extinction[0], 0.0f, 20.0f)) {
            thinfilm.setExtinction(extinction);
            katsuo_top_layer->setThinfilm(thinfilm);
            is_camera_updated = true;
        }
        if (ImGui::SliderFloat("Thickness Scale", &thickness_scale, 0.0f, 1000.0f)) {
            thinfilm.setThicknessScale(thickness_scale);
            katsuo_top_layer->setThinfilm(thinfilm);
            is_camera_updated = true;
        }
        if (ImGui::SliderFloat("TF IOR", &tf_ior, 1.0f, 20.0f)) {
            thinfilm.setTfIor(tf_ior);
            katsuo_top_layer->setThinfilm(thinfilm);
            is_camera_updated = true;
        }

        ImGui::TreePop();
    }
    katsuo_top_layer->copyToDevice();


    // Update bunny 
    //auto bunny = scene.getObject("bunny");
    //auto bunny_mat = dynamic_pointer_cast<Conductor>(bunny->materials[0]);
    //
    //Vec3f ior = bunny_mat->ior();
    //auto tf_thickness_ptr = dynamic_pointer_cast<ConstantTexture>(bunny_mat->thinfilmThickness());
    //Vec3f tf_thickness = tf_thickness_ptr->color();
    //float tf_ior = bunny_mat->thinfilmIOR();
    //Vec3f extinction = bunny_mat->extinction();
    //if (ImGui::TreeNode("Bunny")) {
    //    if (ImGui::SliderFloat3("IOR", &ior[0], 1.0f, 20.0f)) {
    //        bunny_mat->setIOR(ior);
    //        is_camera_updated = true;
    //    }
    //    if (ImGui::SliderFloat("TF IOR", &tf_ior, 1.0f, 20.0f)) {
    //        bunny_mat->setThinfilmIOR(tf_ior);
    //        is_camera_updated = true;
    //    }
    //    if (ImGui::SliderFloat("TF Thickness", &tf_thickness.x(), 0.0f, 1000.0f)) {
    //        tf_thickness_ptr->setColor(tf_thickness);
    //        bunny_mat->setThinfilmThickness(tf_thickness_ptr);
    //        is_camera_updated = true;
    //    }
    //    if (ImGui::SliderFloat3("Extinction", &extinction[0], 0.0f, 20.0f)) {
    //        bunny_mat->setExtinction(extinction);
    //        is_camera_updated = true;
    //    }

    //    ImGui::TreePop();
    //}
    //bunny_mat->copyToDevice();

    //// Update sphere parameters
    //auto sphere = scene.getObject("sphere");
    //auto sphere_mat = dynamic_pointer_cast<Dielectric>(sphere->materials[0]);

    //float ior_sphere = sphere_mat->ior();
    //auto tf_thickness_sphere_ptr = dynamic_pointer_cast<ConstantTexture>(sphere_mat->thinfilmThickness());
    //Vec3f tf_thickness_sphere = tf_thickness_sphere_ptr->color();
    //float tf_ior_sphere = sphere_mat->thinfilmIOR();
    //Vec3f extinction_sphere = sphere_mat->extinction();
    //if (ImGui::TreeNode("Sphere")) {
    //    if (ImGui::SliderFloat("IOR", &ior_sphere, 1.0f, 20.0f)) {
    //        sphere_mat->setIor(ior_sphere);
    //        is_camera_updated = true;
    //    }
    //    if (ImGui::SliderFloat("TF IOR", &tf_ior_sphere, 1.0f, 20.0f)) {
    //        sphere_mat->setThinfilmIOR(tf_ior_sphere);
    //        is_camera_updated = true;
    //    }
    //    if (ImGui::SliderFloat("TF Thickness", &tf_thickness_sphere.x(), 0.0f, 1000.0f)) {
    //        tf_thickness_sphere_ptr->setColor(tf_thickness_sphere);
    //        sphere_mat->setThinfilmThickness(tf_thickness_sphere_ptr);
    //        is_camera_updated = true;
    //    }
    //    if (ImGui::SliderFloat3("Extinction", &extinction_sphere[0], 0.0f, 20.0f)) {
    //        sphere_mat->setExtinction(extinction_sphere);
    //        is_camera_updated = true;
    //    }
    //    ImGui::TreePop();
    //}
    //sphere_mat->copyToDevice();

        

    ImGui::End();
    ImGui::Render();

    result_bmp.draw();

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



