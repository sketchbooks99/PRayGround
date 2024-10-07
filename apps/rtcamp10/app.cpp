#include "app.h"

#include <chrono>

#define USE_DENOISER 0

void App::initResultBufferOnDevice() 
{
    params.frame = 0u;

    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();
    normal_bmp.allocateDevicePtr();
    albedo_bmp.allocateDevicePtr();

    params.result_buffer = (Vec4f*)result_bmp.deviceData();
    params.accum_buffer = (Vec4f*)accum_bmp.deviceData();
    params.normal_buffer = (Vec4f*)normal_bmp.deviceData();
    params.albedo_buffer = (Vec4f*)albedo_bmp.deviceData();
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
#if _DEBUG
    string dst = "Debug";
#else
    string dst = "Release";
#endif
    Module module = pipeline.createModuleFromOptixIr(ctx, "ptx/" + dst + "/rtcamp10_generated_kernels.cu.optixir");

    // Initialize bitmap 
    const int32_t width = pgGetWidth();
    const int32_t height = pgGetHeight();
    result_bmp.allocate(PixelFormat::RGBA, width, height);
    accum_bmp.allocate(PixelFormat::RGBA, width, height);
    normal_bmp.allocate(PixelFormat::RGBA, width, height);
    albedo_bmp.allocate(PixelFormat::RGBA, width, height);
    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();
    normal_bmp.allocateDevicePtr();
    albedo_bmp.allocateDevicePtr();

    // Launch parameter initialization
    params.width = width;
    params.height = height;
#if SUBMISSION
    params.samples_per_launch = 64;
#else
    params.samples_per_launch = 8;
#endif
    params.frame = 0u;
    params.max_depth = 10;
    params.result_buffer = (Vec4f*)result_bmp.deviceData();
    params.accum_buffer = (Vec4f*)accum_bmp.deviceData();
    params.normal_buffer = (Vec4f*)normal_bmp.deviceData();
    params.albedo_buffer = (Vec4f*)albedo_bmp.deviceData();

    // Setup scene
    Scene<Camera, NRay>::AccelSettings accel_settings;
    accel_settings.allow_accel_compaction = true;
    accel_settings.allow_accel_update = true;
    scene.setup(accel_settings);

    // Setup camera
    shared_ptr<Camera> camera = make_shared<Camera>();
    camera->setOrigin(0.0f, 5.0f, -125.0f);
    camera->setLookat(0.0f, -80.0f, 0.0f);
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
    //auto envmap_texture = make_shared<FloatBitmapTexture>("resources/christmas_photo_studio_01_4k.exr", bitmap_id);
    auto envmap_texture = make_shared<ConstantTexture>(Vec3f(0.0f), constant_id);
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

    uint32_t light_sample_sphere_id = setupCallable("__direct_callable__sample_light_sphere", "");
    uint32_t light_pdf_sphere_id = setupCallable("__direct_callable__pdf_light_sphere", "");
    uint32_t light_sample_plane_id = setupCallable("__direct_callable__sample_light_plane", "");
    uint32_t light_pdf_plane_id = setupCallable("__direct_callable__pdf_light_plane", "");

    // Light sampling callables


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
    //auto bunny_mat = make_shared<Conductor>(conductor_id, 
    //    /* texture = */ make_shared<ConstantTexture>(Vec3f(0.8f, 0.8f, 0.3f), constant_id), 
    //    /* twosided = */ true, 
    //    Thinfilm(
    //        /* ior = */ Vec3f(1.9f),
    //        /* thickness = */ make_shared<ConstantTexture>(Vec3f(1.0f), constant_id),
    //        /* thickness_scale = */ 550.0f,
    //        /* tf_ior = */ 1.33f,
    //        /* extinction = */ Vec3f(1.5f)
    //    ));

    auto bunny_mat = make_shared<Disney>(disney_id, make_shared<CheckerTexture>(Vec3f(0.8f, 0.3f, 0.3f), Vec3f(0.2f), 10, checker_id));
    bunny_mat->setTwosided(true);
    bunny_mat->setSpecular(0.8f);
    bunny_mat->setMetallic(0.8f);
    bunny_mat->setRoughness(0.2f);

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
    sashimi_plate_mat->setMetallic(0.0f);
    sashimi_plate_mat->setSpecular(0.2f);
    sashimi_plate_mat->setRoughness(0.5f);
    sashimi_plate_mat->setSheen(0.5f);

    auto light1 = make_shared<AreaEmitter>(area_emitter_id, make_shared<ConstantTexture>(Vec3f(1.0f), constant_id), 50.0f);

    // Setup geometries in the scene
    // Floor geometry
    //shared_ptr<Plane> plane = make_shared<Plane>(Vec2f(-200), Vec2f(200));
    //scene.addObject("floor", plane, floor_mat, { plane_prg, plane_shadow_prg }, Matrix4f::translate(0, -100, 0));

    // Sphere geometry
    //shared_ptr<Sphere> sphere = make_shared<Sphere>(10);
    //scene.addObject("sphere", sphere, sphere_mat, { sphere_prg, sphere_shadow_prg }, Matrix4f::translate(-20, -85, 0));

    //shared_ptr<TriangleMesh> bunny = make_shared<TriangleMesh>("resources/uv_bunny.obj");
    //scene.addObject("bunny", bunny, bunny_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(20, -95, 0) * Matrix4f::scale(100));

    shared_ptr<TriangleMesh> katsuo = make_shared<TriangleMesh>("resources/Katsuo.obj");
    scene.addObject("katsuo", katsuo, katsuo_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(33, -92, 0) * Matrix4f::scale(10));

    float x_range = 80.0f;
    for (int i = 0; i < 5; i++) {
        float x = 66.0f / 2.0f - (float)(i + 1) * (x_range / 6.0f);
        scene.addObject("katsuo" + to_string(i), katsuo, katsuo_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(x, -90, 2)* Matrix4f::rotate(math::pi / 9.0f, Vec3f(0, 0, 1)) * Matrix4f::scale(10));
    }

    shared_ptr<TriangleMesh> table = make_shared<TriangleMesh>("resources/Table.obj");
    scene.addObject("table", table, table_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(0, -90, 0)* Matrix4f::scale(10));

    shared_ptr<TriangleMesh> sashimi_plate = make_shared<TriangleMesh>("resources/Sashimi plate.obj");
    sashimi_plate->calculateNormalFlat();
    scene.addObject("sashimi plate", sashimi_plate, sashimi_plate_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(0, -90, 0)* Matrix4f::scale(10));


    //shared_ptr<Sphere> light1_geom = make_shared<Sphere>(5);
    //scene.addLight("light1", light1_geom, light1, { sphere_prg, sphere_shadow_prg }, Matrix4f::translate(50, 20, 30));
    //scene.addLight("light2", light1_geom, light1, { sphere_prg, sphere_shadow_prg }, Matrix4f::translate(-50, 20, 30));
    //scene.addLight("light3", light1_geom, light1, { sphere_prg, sphere_shadow_prg }, Matrix4f::translate(-50, 20, -30));
    //scene.addLight("light4", light1_geom, light1, { sphere_prg, sphere_shadow_prg }, Matrix4f::translate(50, 20, -30));
    shared_ptr<Plane> light_geom = make_shared<Plane>(Vec2f(-50, -20), Vec2f(50, 20));
    scene.addLight("light", light_geom, light1, { plane_prg, plane_shadow_prg }, Matrix4f::translate(0, 20, 0));


#if USE_DENOISER
    // Denoiser setting
    denoise_data.width = width;
    denoise_data.height = height;
    denoise_data.outputs.push_back(new float[denoise_data.width * denoise_data.height * 4]);
    denoise_data.color = result_bmp.deviceData();
    denoise_data.albedo = albedo_bmp.deviceData();
    denoise_data.normal = normal_bmp.deviceData();
    denoiser.init(ctx, denoise_data, 0, 0, false, false);
#endif

    // Setup context
    CUDA_CHECK(cudaStreamCreate(&stream));
    scene.copyDataToDevice();

    std::vector<LightInfo> lights;
    for (auto name : scene.lightNames()) {
        auto light = scene.getLight(name);
        lights.push_back({
            .shape_data = light->shape->devicePtr(),
            .objToWorld = light->instance.transform(),
            .worldToObj = light->instance.transform().inverse(),
            .sample_id = light_sample_plane_id,
            .pdf_id = light_pdf_plane_id,
            .twosided = true,
            .surface_info = light->emitters[0]->surfaceInfoDevicePtr()
        });
    }
    CUDABuffer<LightInfo> d_lights;
    d_lights.copyToDevice(lights);
    params.lights = d_lights.deviceData();
    params.num_lights = lights.size();

    scene.buildAccel(ctx, stream);
    scene.buildSBT();
    pipeline.create(ctx);

    params.handle = scene.accelHandle();

#if SUBMISSION
    double elapsed_seconds = 0.0;
    int32_t frame = 0;

    float r = 125.0f;
    while (elapsed_seconds < 256.0) {
        elapsed_seconds = (double)duration_cast<milliseconds>(system_clock::now() - start).count() / 1000;
        if (elapsed_seconds > 256.0) break;

        scene.launchRay(ctx, pipeline, params, stream, result_bmp.width(), result_bmp.height(), 1);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_SYNC_CHECK();

        result_bmp.copyFromDevice();
        normal_bmp.copyFromDevice();
        albedo_bmp.copyFromDevice();

        denoise_data.color = result_bmp.deviceData();
        denoise_data.normal = normal_bmp.deviceData();
        denoise_data.albedo = albedo_bmp.deviceData();

        denoiser.update(denoise_data);
        denoiser.run();

        denoiser.copyFromDevice();

        string filename = std::format("{:03d}.png", frame);
        filesystem::path filepath = pgPathJoin(pgGetExecutableDir(), filename);
        denoiser.write(denoise_data, filepath);

        const float x = sinf(elapsed_seconds * 0.1f) * r;
        const float z = cosf(elapsed_seconds * 0.1f) * r;
        camera->setOrigin(x, -5.0f, z);
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
#if USE_DENOISER
    normal_bmp.copyFromDevice();
    albedo_bmp.copyFromDevice();

    denoise_data.color = result_bmp.deviceData();
    denoise_data.normal = normal_bmp.deviceData();
    denoise_data.albedo = albedo_bmp.deviceData();

    denoiser.update(denoise_data);
    denoiser.run();

    denoiser.copyFromDevice();
#endif

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

    ImGui::Text("Camera info:");
    ImGui::Text("Origin,: (%.2f, %.2f, %.2f)", scene.camera()->origin().x(), scene.camera()->origin().y(), scene.camera()->origin().z());
    ImGui::Text("Lookat: (%.2f, %.2f, %.2f)", scene.camera()->lookat().x(), scene.camera()->lookat().y(), scene.camera()->lookat().z());
    ImGui::Text("Up: (%.2f, %.2f, %.2f)", scene.camera()->up().x(), scene.camera()->up().y(), scene.camera()->up().z());

    ImGui::End();
    ImGui::Render();

#if USE_DENOISER
    denoiser.draw(denoise_data, 0, 0);
#else
    result_bmp.draw();
#endif

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



