#include "app.h"

#include <chrono>

#define USE_DENOISER 1

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
    pipeline.setTraceDepth(15);

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

    // Video will be 5 seconds at 24fps = 120 frames
    cam_pos_keypoints = {
        { Vec3f(105.0f, -33.0f, 148.0f), 0.0f}, // lemon start
        { Vec3f(96.0f, -67.0f, 78.0f), 2.0f}, // lemon
        { Vec3f(54.0f, -73.0f, -123.4f), 2.00001f}, // chopstick start
        { Vec3f(67.0f, -78.0f, -73.0f), 4.4f }, // chopstick end
        { Vec3f(-87.5f, -29.0f, -42.0f), 4.40001f }, // beer start
        { Vec3f(-60.0f, -51.0f, -7.0f), 7.0f }, // beer end
        { Vec3f(0.0f, -4.0f, -200.0f), 7.00001f}, // all start
        { Vec3f(0.0f, -26.0f, -145.0f), 9.0f }, // all end
        { Vec3f(0.0f, -26.0f, -145.0f), 10.0f } // all end
    };
    cam_look_keypoints = {
        { Vec3f(40.0f, -92.0f, 52.0f), 0.0f}, // lemon start
        { Vec3f(40.0f, -92.0f, 52.0f), 2.0f}, // lemon start
        { Vec3f(2.2f, -112.0f, -23.8f), 2.00001f}, // chopstick start
        { Vec3f(0.0f, -108.0f, -28.0f), 4.4f }, // chopstick end
        { Vec3f(-60.0f, -60.0f, 25.0f), 4.40001f}, // beer start
        { Vec3f(-60.0f, -60.0f, 25.0f), 7.0f}, // beer end
        { Vec3f(0.0f, -90.0f, 15.0f), 7.00001f }, // all start
        { Vec3f(0.0f, -90.0f, 15.0f), 9.0f }, // all end
        { Vec3f(0.0f, -90.0f, 15.0f), 10.0f } // all end
    };

    // Setup camera
    shared_ptr<Camera> camera = make_shared<Camera>();
    camera->setOrigin(105.0f, -33.0f, 148.0f);
    camera->setLookat(40.0f, -92.0f, 52.0f);
    //camera->setOrigin(0.0f, -26.0f, -145.0f);
    //camera->setLookat(0.0f, -90.0f, 15.0f);
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
    auto gradient_id = setupCallable("__direct_callable__gradient", "");

    // Miss program
    ProgramGroup miss = pipeline.createMissProgram(ctx, module, "__miss__envmap");
    ProgramGroup miss_shadow = pipeline.createMissProgram(ctx, module, "__miss__shadow");
    scene.bindMissPrograms({ miss, miss_shadow });
    //std::vector<Keypoint<Vec3f>> envmap_keypoints = {
    //    { Vec3f(47.0f / 255.0f, 101.0f / 255.0f, 190.0f / 255.0f), 0.0f},
    //    { Vec3f(86.0f / 255.0f, 157.0f / 255.0f, 225.0f / 255.0f), 0.3f },
    //    { Vec3f(208.0f / 255.0f, 228.0f / 255.0f, 243.0f / 255.0f), 0.49f },
    //    { Vec3f(208.0f / 255.0f, 228.0f / 255.0f, 243.0f / 255.0f), 0.50f },
    //    { Vec3f(252.0f / 255.0f, 175.0f / 255.0f, 127.0f / 255.0f), 0.51f },
    //    { Vec3f(49.0f / 255.0f, 89.0f / 255.0f, 143.0f / 255.0f), 0.63f },
    //    { Vec3f(26.0f / 255.0f, 68.0f / 255.0f, 114.0f / 255.0f), 1.0f },
    //};
    //auto envmap_texture = make_shared<GradientTexture>(envmap_keypoints, gradient_id, EaseType::Linear, Vec2f(0, 1));
    auto envmap_texture = make_shared<FloatBitmapTexture>("resources/drackenstein_quarry_4k.exr", bitmap_id);
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
    // Point cloud
    ProgramGroup pcd_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__custom", "__intersection__point_cloud");
    ProgramGroup pcd_shadow_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__shadow", "__intersection__point_cloud");

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
    SurfaceCallableID isotropic_id = {
        setupCallable("__direct_callable__sample_isotropic", ""),
        setupCallable("__direct_callable__bsdf_isotropic", ""),
        setupCallable("__direct_callable__pdf_isotropic", "")
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

    auto bunny_mat = make_shared<Disney>(disney_id, make_shared<CheckerTexture>(Vec3f(0.8f, 0.3f, 0.3f), Vec3f(0.2f), 10, checker_id));
    bunny_mat->setTwosided(true);
    bunny_mat->setSpecular(0.8f);
    bunny_mat->setMetallic(0.8f);
    bunny_mat->setRoughness(0.2f);

    //auto katsuo_base = make_shared<Diffuse>(diffuse_id, make_shared<BitmapTexture>("resources/Katsuo Color.png", bitmap_id));
    auto katsuo_base = make_shared<Disney>(disney_id, make_shared<BitmapTexture>("resources/Katsuo Color.png", bitmap_id));
    katsuo_base->setSubsurface(0.8f);
    katsuo_base->setMetallic(0.0f);
    katsuo_base->setSpecular(0.4f);
    katsuo_base->setRoughness(0.6f);
    katsuo_base->setSheen(0.5f);
    auto katsuo_oil = make_shared<Dielectric>(dielectric_id,
        make_shared<ConstantTexture>(Vec3f(0.9f, 0.9f, 0.8f), constant_id),
        1.5f, 0.0f, Sellmeier::None,
        Thinfilm(Vec3f(3.5f, 4.3f, 3.1f), make_shared<BitmapTexture>("resources/Katsuo thinfilm thickness.png", bitmap_id), 20.0f, 1.6f, Vec3f(3.0f, 6.0f, 2.0f)));
    std::vector <std::shared_ptr<Material>> layers = { katsuo_oil, katsuo_base };
    auto katsuo_mat = make_shared<Layered>(layered_id, layers);
    katsuo_mat->setBumpmap(make_shared<BitmapTexture>("resources/Katsuo Normal.png", bitmap_id));

    // Table
    auto table_mat = make_shared<Disney>(disney_id, make_shared<BitmapTexture>("resources/Table Color.png", bitmap_id));
    table_mat->setSubsurface(0.0f);
    table_mat->setSpecular(0.5f);
    table_mat->setRoughness(0.5f);
    table_mat->setMetallic(0.0f);
    table_mat->setSheenTint(0.5f);

    // Sashimi plate
    auto sashimi_plate_mat = make_shared<Disney>(disney_id, make_shared<ConstantTexture>(Vec3f(1.0f), constant_id));
    sashimi_plate_mat->setSubsurface(0.0f);
    sashimi_plate_mat->setMetallic(0.0f);
    sashimi_plate_mat->setSpecular(0.2f);
    sashimi_plate_mat->setRoughness(0.5f);
    sashimi_plate_mat->setSheen(0.5f);

    // Lemon 
    auto lemon_mat = make_shared<Disney>(disney_id, make_shared<BitmapTexture>("resources/Lemon Color.png", bitmap_id));
    lemon_mat->setSubsurface(0.5f);
    lemon_mat->setMetallic(0.0f);
    lemon_mat->setSpecular(0.8f);
    lemon_mat->setRoughness(0.4f);
    lemon_mat->setSheenTint(0.5f);
    lemon_mat->setBumpmap(make_shared<BitmapTexture>("resources/Lemon Normal.png", bitmap_id));

    auto dragon_mat = make_shared<Conductor>(conductor_id, make_shared<ConstantTexture>(Vec3f(0.8f, 0.8f, 0.3f), constant_id), true, 
        Thinfilm(Vec3f(1.9f), make_shared<ConstantTexture>(Vec3f(1.0f), constant_id), 550.0f, 1.33f, Vec3f(1.5f)));

    auto chopstick_mat = make_shared<Disney>(disney_id, make_shared<BitmapTexture>("resources/ChopStick Color.png", bitmap_id));
    table_mat->setSubsurface(0.3f);
    table_mat->setSpecular(0.5f);
    table_mat->setRoughness(0.5f);
    table_mat->setMetallic(0.0f);
    table_mat->setSheenTint(0.5f);

    // Beer
    auto beer_glass_mat = make_shared<Dielectric>(dielectric_id, make_shared<ConstantTexture>(Vec3f(1.0f), constant_id), 1.5f, 0.0f, Sellmeier::None);
    auto beer_mat = make_shared<Dielectric>(dielectric_id, make_shared<ConstantTexture>(Vec3f(0.93f, 0.7f, 0.13f), constant_id), 1.33f, 0.05f, Sellmeier::None);

    auto beer_awa_mat = make_shared<Diffuse>(diffuse_id, make_shared<ConstantTexture>(Vec3f(1.0f), constant_id));
    auto beer_label_mat = make_shared<Disney>(disney_id, make_shared<BitmapTexture>("resources/BeerLabel.png", bitmap_id));
    beer_label_mat->setMetallic(0.3f);
    beer_label_mat->setSpecular(0.6f);
    beer_label_mat->setRoughness(0.2f);

    auto beer_awa_float_mat = make_shared<Isotropic>(isotropic_id, Vec3f(1.0f));

    auto light1 = make_shared<AreaEmitter>(area_emitter_id, make_shared<ConstantTexture>(Vec3f(1.0f), constant_id), 50.0f);

    // Setup geometries in the scene
    shared_ptr<TriangleMesh> katsuo = make_shared<TriangleMesh>("resources/Katsuo.obj");
    scene.addObject("katsuo", katsuo, katsuo_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(33, -92, 0) * Matrix4f::scale(10));

    float x_range = 80.0f;
    for (int i = 0; i < 5; i++) {
        float x = 66.0f / 2.0f - (float)(i + 1) * (x_range / 6.0f);
        scene.addObject("katsuo" + to_string(i), katsuo, katsuo_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(x, -90, 2)* Matrix4f::rotate(math::pi / 9.0f, Vec3f(0, 0, 1)) * Matrix4f::scale(10));
    }

    shared_ptr<TriangleMesh> sashimi_plate = make_shared<TriangleMesh>("resources/Sashimi plate.obj");
    sashimi_plate->calculateNormalFlat();
    scene.addObject("sashimi plate", sashimi_plate, sashimi_plate_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(0, -90, 0) * Matrix4f::scale(10));

    shared_ptr<TriangleMesh> table = make_shared<TriangleMesh>("resources/Table.obj");
    scene.addObject("table", table, table_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(0, -104, 0)* Matrix4f::scale(25));

    shared_ptr<TriangleMesh> lemon = make_shared<TriangleMesh>("resources/Lemon.obj");
    lemon->calculateNormalSmooth();
    scene.addObject("lemon1", lemon, lemon_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(58, -88, 55) * Matrix4f::rotate(math::pi / 2.0f, Vec3f(1, 1, 1)) * Matrix4f::scale(10));
    scene.addObject("lemon2", lemon, lemon_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(18, -88, 48) * Matrix4f::rotate(math::pi / 2.0f, Vec3f(1, 0, 0)) * Matrix4f::scale(10));
    scene.addObject("lemon3", lemon, lemon_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(35, -88, 62) * Matrix4f::rotate(math::pi / 2.0f, Vec3f(1, 0, 1)) * Matrix4f::scale(10));

    shared_ptr<TriangleMesh> dragon = make_shared<TriangleMesh>("resources/dragon.obj");
    scene.addObject("dragon", dragon, dragon_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(30, -91, -54) * Matrix4f::scale(20));

    shared_ptr<TriangleMesh> chopstick = make_shared<TriangleMesh>("resources/ChopStick.obj");
    scene.addObject("chopstick", chopstick, chopstick_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(0, -89, -50) * Matrix4f::rotate(math::pi / 30.0f, Vec3f(0,0,1)) * Matrix4f::scale(5));

    // beer
    auto beer_x = -60;
    auto beer_y = -70;
    auto beer_z = 50;
    auto beer_scale = 13;
    shared_ptr<TriangleMesh> beer_glass = make_shared<TriangleMesh>("resources/BeerGlass.obj");
    scene.addObject("beer glass", beer_glass, beer_glass_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(beer_x, beer_y, beer_z) * Matrix4f::scale(beer_scale));
    shared_ptr<TriangleMesh> beer = make_shared<TriangleMesh>("resources/Beer.obj");
    scene.addObject("beer", beer, beer_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(beer_x, beer_y, beer_z) * Matrix4f::scale(beer_scale));
    shared_ptr<TriangleMesh> beer_awa = make_shared<TriangleMesh>("resources/BeerAwa.obj");
    scene.addObject("beer awa", beer_awa, beer_awa_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(beer_x, beer_y, beer_z) * Matrix4f::scale(beer_scale));
    shared_ptr<TriangleMesh> beer_label = make_shared<TriangleMesh>("resources/BeerLabel.obj");
    scene.addObject("beer label", beer_label, beer_label_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(beer_x, beer_y, beer_z) * Matrix4f::scale(beer_scale));

    // Awa point cloud
    seed = tea<4>(0, 0);
    awa_pcd_points.resize(800);
    for (auto &point : awa_pcd_points) {
        float r = rnd(seed) * 26.0f - 13.0f;
        float theta = rnd(seed) * 2.0f * math::pi;
        float phi = rnd(seed) * 2.0f * math::pi;
        point.point = Vec3f(r * sinf(theta) * cosf(phi), 5 * cos(theta), r * sinf(theta) * sinf(phi)) + Vec3f(beer_x, beer_y + 15, beer_z);
        point.radius = 0.07f;
    }
    awa_pcd = make_shared<PointCloud>(awa_pcd_points);
    scene.addObject("beer awa pcd", awa_pcd, beer_awa_mat, { pcd_prg, pcd_shadow_prg }, Matrix4f::identity());

    shared_ptr<Plane> light_geom = make_shared<Plane>(Vec2f(-50, -20), Vec2f(50, 20));
    scene.addLight("light", light_geom, light1, { plane_prg, plane_shadow_prg }, Matrix4f::translate(0, 20, 0));


#if USE_DENOISER
    // Denoiser setting
    denoise_data.width = width;
    denoise_data.height = height;
    denoise_data.outputs.push_back(new float[denoise_data.width * denoise_data.height * 4]);
    denoise_data.color = result_bmp.deviceData();
    denoise_data.albedo = albedo_bmp.deviceData();
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
    static constexpr int32_t fps = 12;
    static constexpr int32_t MAX_FRAME = 120;
    static constexpr int32_t NUM_SAMPLES = 96;
    static constexpr EaseType ease = EaseType::InOutQuad;
    float frame_time = 0.0f;

    while (elapsed_seconds < 256.0f && frame < MAX_FRAME) {
        elapsed_seconds = (double)duration_cast<milliseconds>(system_clock::now() - start).count() / 1000;
        if (elapsed_seconds > 256.0f) break;

        params.samples_per_launch = 8;
        int32_t n_iter = NUM_SAMPLES / params.samples_per_launch;
        for (int i = 0; i < n_iter; i++) {
            scene.launchRay(ctx, pipeline, params, stream, result_bmp.width(), result_bmp.height(), 1);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_SYNC_CHECK();

            result_bmp.copyFromDevice();
            albedo_bmp.copyFromDevice();

            denoise_data.color = result_bmp.deviceData();
            denoise_data.albedo = albedo_bmp.deviceData();

            denoiser.update(denoise_data);
            denoiser.run();
            params.frame++;
        }

        denoiser.copyFromDevice();

        string filename = std::format("{:03d}.png", frame);
        filesystem::path filepath = pgPathJoin(pgGetExecutableDir(), filename);
        denoiser.write(denoise_data, filepath);

        frame++;
        auto frame_interval = 1.0f / fps;
        frame_time += frame_interval;

        float velocity = -25.0f;
        for (auto& point : awa_pcd_points) {
            Vec3f new_point = point.point + Vec3f(0, velocity * frame_interval, 0);
            if (new_point.y() > beer_y - 10) {
                float r = rnd(seed) * 28.0f - 14.0f;
                float theta = rnd(seed) * 2.0f * math::pi;
                float phi = rnd(seed) * 2.0f * math::pi;
                point.point = Vec3f(r * sinf(theta) * cosf(phi), 3 * cos(theta), r * sinf(theta) * sinf(phi)) + Vec3f(beer_x, beer_y + 20, beer_z);
            }
        }
        awa_pcd->updatePoints(awa_pcd_points);
        awa_pcd->copyToDevice();
        scene.updateObjectGAS("beer awa pcd", ctx, stream);
        scene.updateSBT(+(SBTRecordType::Hitgroup));
        scene.updateAccel(ctx, stream);

        // Find two keypoints to interpolate
        float min_t = 0.0f; float max_t = 10.0f;
        bool keypoint_found = false;
        for (int i = 0; i < cam_pos_keypoints.size() - 1; i++) {
            min_t = fminf(min_t, cam_pos_keypoints[i].t);
            max_t = fmaxf(max_t, cam_pos_keypoints[i].t);
            if (cam_pos_keypoints[i].t <= frame_time && frame_time < cam_pos_keypoints[i + 1].t) {
                Vec3f pos = Keypoint<Vec3f>::ease(cam_pos_keypoints[i], cam_pos_keypoints[i + 1], frame_time, ease);
                Vec3f look = Keypoint<Vec3f>::ease(cam_look_keypoints[i], cam_look_keypoints[i + 1], frame_time, ease);
                scene.camera()->setOrigin(pos);
                scene.camera()->setLookat(look);
                keypoint_found = true;
                break;
            }
        }

        if (!keypoint_found) {
            min_t = fminf(min_t, cam_pos_keypoints[cam_pos_keypoints.size() - 1].t);
            max_t = fmaxf(max_t, cam_pos_keypoints[cam_pos_keypoints.size() - 1].t);

            if (frame_time < min_t) {
                scene.camera()->setOrigin(cam_pos_keypoints[0].value);
                scene.camera()->setLookat(cam_look_keypoints[0].value);
            }
            else if (frame_time > max_t) {
                scene.camera()->setOrigin(cam_pos_keypoints[cam_pos_keypoints.size() - 1].value);
                scene.camera()->setLookat(cam_look_keypoints[cam_look_keypoints.size() - 1].value);
            }
        }
        is_camera_updated = true;

        cout << elapsed_seconds << "s elapsed, " << frame << " frames rendered" << endl;

        handleCameraUpdate();
    }

    pgExit();
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
    albedo_bmp.copyFromDevice();
    normal_bmp.copyFromDevice();
#if USE_DENOISER

    denoise_data.color = result_bmp.deviceData();
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
        if (ImGui::SliderFloat("Thickness Scale", &thickness_scale, 0.0f, 2000.0f)) {
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
    ImGui::Text("Num samples: %d", params.samples_per_launch * params.frame);

    
    // Update camera
    Vec3f origin = scene.camera()->origin();
    Vec3f lookat = scene.camera()->lookat();
    if (ImGui::InputFloat3("Origin", &origin[0])) {
        scene.camera()->setOrigin(origin);
        is_camera_updated = true;
    }
    if (ImGui::InputFloat3("Lookat", &lookat[0])) {
        scene.camera()->setLookat(lookat);
        is_camera_updated = true;
    }

    ImGui::End();
    ImGui::Render();

    int w = pgGetWidth();
    int h = pgGetHeight();

    result_bmp.draw(0, 0, w / 2, h / 2);
    normal_bmp.draw(w / 2, 0, w / 2, h / 2);
    albedo_bmp.draw(0, h / 2, w / 2, h / 2);

#if USE_DENOISER
    denoiser.draw(denoise_data, w / 2, h / 2, w / 2, h / 2);
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
    if (key == Key::S) {
        denoiser.write(denoise_data, pgPathJoin(pgGetExecutableDir(), "result.png"));
    }
}

// ------------------------------------------------------------------
void App::keyReleased(int key)
{

}



