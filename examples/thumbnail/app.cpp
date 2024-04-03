#include "app.h"

#define INTERACTIVE 0

static void streamProgress(int frame, int max_frame, float elapsed_time, int bar_length)
{
    cout << "\rRendering: [";
    int progress = static_cast<int>(((float)(frame) / max_frame) * bar_length);
    for (int i = 0; i < progress; i++)
        cout << "+";
    for (int i = progress; i < bar_length; i++)
        cout << " ";
    cout << "]";

    cout << " [" << fixed << setprecision(2) << elapsed_time << "s]";

    float percent = (float)(frame) / max_frame;
    cout << " (" << fixed << setprecision(2) << (float)(percent * 100.0f) << "%, ";
    cout << "Samples: " << frame << " / " << max_frame << ")" << flush;
    if (frame == max_frame)
        cout << endl;
}

void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<Vec4u*>(result_bitmap.deviceData());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bitmap.deviceData());

    CUDA_SYNC_CHECK();
}

void App::handleCameraUpdate()
{
    if (!camera_update)
        return;
    camera_update = false;

    RaygenRecord* rg_record = reinterpret_cast<RaygenRecord*>(sbt.deviceRaygenRecordPtr());
    RaygenData rg_data;
    rg_data.camera = camera.getData();

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(&rg_record->data),
        &rg_data, sizeof(RaygenData),
        cudaMemcpyHostToDevice
    ));

    initResultBufferOnDevice();
}

// ----------------------------------------------------------------
void App::setup()
{
    // Initialize CUDA
    stream = 0;
    CUDA_CHECK(cudaFree(0));
    // Initialize OptixDeviceContext
    OPTIX_CHECK(optixInit());
    context.disableValidation();
    context.create();

    // Initialize instance acceleration structure
    scene_ias = InstanceAccel{ InstanceAccel::Type::Instances };

    // Pipeline settings
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(5);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(6);

    // Create modules from cuda source file 
    Module raygen_module, miss_module, hitgroups_module, textures_module, surfaces_module;
    raygen_module = pipeline.createModuleFromCudaFile(context, "cuda/raygen.cu");
    miss_module = pipeline.createModuleFromCudaFile(context, "cuda/miss.cu");
    hitgroups_module = pipeline.createModuleFromCudaFile(context, "cuda/hitgroups.cu");
    textures_module = pipeline.createModuleFromCudaFile(context, "cuda/textures.cu");
    surfaces_module = pipeline.createModuleFromCudaFile(context, "cuda/surfaces.cu");

    // Initialize bitmaps to store rendered results
    result_bitmap.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());
    accum_bitmap.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());

    // Configuration of launch parameters
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;
    params.white = 1.0f;

    initResultBufferOnDevice();

    // Camera settings
    camera.setOrigin(-47.7f, 10.0f, 64.0f);
    camera.setLookat(-7, -4.5, 12.3);
    camera.setUp(0.0f, 1.0f, 0.0f);
    camera.setFarClip(5000);
    camera.setFov(40.0f);
    camera.setAspect(static_cast<float>(params.width) / params.height);
    camera.setAperture(2.0f);
    camera.setFocusDistance(60);
    camera.enableTracking(pgGetCurrentWindow());

    // Raygen program
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, raygen_module, "__raygen__lens");
    // Shader binding table data for raygen program
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera = camera.getData();
    sbt.setRaygenRecord(raygen_record);

    auto setupCallable = [&](const Module& module, const std::string& dc, const std::string& cc)
    {
        EmptyRecord callable_record = {};
        auto [prg, id] = pipeline.createCallablesProgram(context, module, dc, cc);
        prg.recordPackHeader(&callable_record);
        sbt.addCallablesRecord(callable_record);
        return id;
    };

    // Callable programs for textures
    uint32_t constant_prg_id = setupCallable(textures_module, DC_FUNC_TEXT("constant"), "");
    uint32_t checker_prg_id = setupCallable(textures_module, DC_FUNC_TEXT("checker"), "");
    uint32_t bitmap_prg_id = setupCallable(textures_module, DC_FUNC_TEXT("bitmap"), "");

    // Callable programs for surfaces 
    // Diffuse
    uint32_t diffuse_sample_bsdf_prg_id = setupCallable(surfaces_module, DC_FUNC_TEXT("sample_diffuse"), CC_FUNC_TEXT("bsdf_diffuse"));
    uint32_t diffuse_pdf_prg_id = setupCallable(surfaces_module, DC_FUNC_TEXT("pdf_diffuse"), "");
    // Conductor
    uint32_t conductor_sample_bsdf_prg_id = setupCallable(surfaces_module, DC_FUNC_TEXT("sample_conductor"), CC_FUNC_TEXT("bsdf_conductor"));
    uint32_t conductor_pdf_prg_id = setupCallable(surfaces_module, DC_FUNC_TEXT("pdf_conductor"), "");
    // Dielectric
    uint32_t dielectric_sample_bsdf_prg_id = setupCallable(surfaces_module, DC_FUNC_TEXT("sample_dielectric"), CC_FUNC_TEXT("bsdf_dielectric"));
    uint32_t dielectric_pdf_prg_id = setupCallable(surfaces_module, DC_FUNC_TEXT("pdf_dielectric"), "");
    // Disney
    uint32_t disney_sample_bsdf_prg_id = setupCallable(surfaces_module, DC_FUNC_TEXT("sample_disney"), CC_FUNC_TEXT("bsdf_disney"));
    uint32_t disney_pdf_prg_id = setupCallable(surfaces_module, DC_FUNC_TEXT("pdf_disney"), "");
    // AreaEmitter
    uint32_t area_emitter_prg_id = setupCallable(surfaces_module, DC_FUNC_TEXT("area_emitter"), "");

    SurfaceCallableID diffuse_id = { diffuse_sample_bsdf_prg_id, diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    SurfaceCallableID conductor_id = { conductor_sample_bsdf_prg_id, conductor_sample_bsdf_prg_id, conductor_pdf_prg_id };
    SurfaceCallableID dielectric_id = { dielectric_sample_bsdf_prg_id, dielectric_sample_bsdf_prg_id, dielectric_pdf_prg_id };
    SurfaceCallableID disney_id = { disney_sample_bsdf_prg_id, disney_sample_bsdf_prg_id, disney_pdf_prg_id };
    SurfaceCallableID area_emitter_id = { area_emitter_prg_id, area_emitter_prg_id, area_emitter_prg_id };

    // Callable program for direct sampling of area emitter
    uint32_t plane_sample_pdf_prg_id = setupCallable(hitgroups_module, DC_FUNC_TEXT("rnd_sample_plane"), CC_FUNC_TEXT("pdf_plane"));

    textures.emplace("env", new FloatBitmapTexture("resources/image/christmas_photo_studio_01_4k.exr", bitmap_prg_id));

    env = EnvironmentEmitter{ textures.at("env") };
    env.copyToDevice();

    // Miss program
    ProgramGroup miss_prg = pipeline.createMissProgram(context, miss_module, MS_FUNC_TEXT("envmap"));
    // Shader binding table data for miss program
    MissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env.devicePtr();
    sbt.setMissRecord({ miss_record });

    // Hitgroup program
    // Plane
    auto plane_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_TEXT("plane"), IS_FUNC_TEXT("plane"));
    auto plane_alpha_discard_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_TEXT("plane"), IS_FUNC_TEXT("plane"), AH_FUNC_TEXT("alpha_discard"));
    // Sphere
    auto sphere_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_TEXT("sphere"), IS_FUNC_TEXT("sphere"));
    auto sphere_alpha_discard_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_TEXT("sphere"), IS_FUNC_TEXT("sphere"), AH_FUNC_TEXT("alpha_discard"));
    // Box
    auto box_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_TEXT("box"), IS_FUNC_TEXT("box"));
    // Cylinder
    auto cylinder_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_TEXT("cylinder"), IS_FUNC_TEXT("cylinder"));
    // Triangle mesh
    auto mesh_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_TEXT("mesh"));

    uint32_t sbt_idx = 0;
    uint32_t sbt_offset = 0;
    uint32_t instance_id = 0;

    using SurfaceP = variant<shared_ptr<Material>, shared_ptr<AreaEmitter>>;
    auto addHitgroupRecord = [&](
        ProgramGroup& prg, shared_ptr<Shape> shape, SurfaceP surface, shared_ptr<Texture> alpha_texture = nullptr)
    {
        const bool is_mat = holds_alternative<shared_ptr<Material>>(surface);
        if (alpha_texture) alpha_texture->copyToDevice();

        // Copy data to GPU
        shape->copyToDevice();
        shape->setSbtIndex(sbt_idx);
        if (is_mat)
            std::get<shared_ptr<Material>>(surface)->copyToDevice();
        else
            std::get<shared_ptr<AreaEmitter>>(surface)->copyToDevice();

        // Register data to shader binding table
        HitgroupRecord record;
        prg.recordPackHeader(&record);
        record.data =
        {
            .shape_data = shape->devicePtr(),
            .surface_info =
            {
                .data = is_mat ? std::get<shared_ptr<Material>>(surface)->devicePtr() : std::get<shared_ptr<AreaEmitter>>(surface)->devicePtr(),
                .callable_id = is_mat ? std::get<shared_ptr<Material>>(surface)->surfaceCallableID() : std::get<shared_ptr<AreaEmitter>>(surface)->surfaceCallableID(),
                .type = is_mat ? std::get<shared_ptr<Material>>(surface)->surfaceType() : SurfaceType::AreaEmitter,
            },
            .alpha_texture = alpha_texture ? alpha_texture->getData() : Texture::Data{ nullptr, bitmap_prg_id }
        };

        sbt.addHitgroupRecord({ record });
        sbt_idx++;
    };

    auto createGAS = [&](shared_ptr<Shape> shape, const Matrix4f& transform, uint32_t num_sbt = 1)
    {
        // Build GAS and add it to IAS
        ShapeInstance instance{ shape->type(), shape, transform };
        instance.allowCompaction();
        instance.buildAccel(context, stream);
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);

        scene_ias.addInstance(instance);

        instance_id++;
        sbt_offset += SBT::NRay * num_sbt;
    };

    auto setupObject = [&](ProgramGroup& prg, shared_ptr<Shape> shape, shared_ptr<Material> material, const Matrix4f& transform, shared_ptr<Texture> alpha_texture = nullptr)
    {
        addHitgroupRecord(prg, shape, material, alpha_texture);
        createGAS(shape, transform);
    };

    vector<AreaEmitterInfo> area_emitter_infos;
    auto setupAreaEmitter = [&](
        ProgramGroup& prg,
        shared_ptr<Shape> shape,
        shared_ptr<AreaEmitter> area, Matrix4f transform,
        uint32_t sample_pdf_id,
        shared_ptr<Texture> alpha_texture = nullptr
        )
    {
        ASSERT(dynamic_pointer_cast<Plane>(shape) || dynamic_pointer_cast<Sphere>(shape), "The shape of area emitter must be a plane or sphere.");

        addHitgroupRecord(prg, shape, area, alpha_texture);
        createGAS(shape, transform);

        AreaEmitterInfo area_emitter_info =
        {
            .shape_data = shape->devicePtr(),
            .objToWorld = transform,
            .worldToObj = transform.inverse(),
            .sample_id = sample_pdf_id,
            .pdf_id = sample_pdf_id
        };
        area_emitter_infos.push_back(area_emitter_info);
    };

    // Scene ==========================================================================
    auto logo_alpha = make_shared<BitmapTexture>("resources/image/PRayGround_white.png", bitmap_prg_id);
    auto logo_aspect = (float)logo_alpha->width() / logo_alpha->height();
    textures.emplace("logo_alpha", logo_alpha);
    textures.emplace("white", new ConstantTexture(make_float3(1.0f), constant_prg_id));
    textures.emplace("black", new ConstantTexture(make_float3(0.0f), constant_prg_id));
    textures.emplace("bright_gray", new ConstantTexture(make_float3(0.6f), constant_prg_id));
    textures.emplace("orange", new ConstantTexture(make_float3(0.8f, 0.7f, 0.3f), constant_prg_id));
    textures.emplace("green", new ConstantTexture(make_float3(0.05f, 0.8f, 0.6f), constant_prg_id));
    textures.emplace("yellow", new ConstantTexture(make_float3(0.8f, 0.8f, 0.05f), constant_prg_id));
    textures.emplace("wine_red", new ConstantTexture(make_float3(0.5f, 0.15f, 0.4f), constant_prg_id));
    textures.emplace("checker", new CheckerTexture(make_float3(0.8f), make_float3(0.3f), 100, checker_prg_id));
    textures.emplace("rtRest", new BitmapTexture("examples/rayTracingRestOfYourLife/rtRestOfYourLife.jpg", bitmap_prg_id));
    textures.emplace("brick", new BitmapTexture("resources/image/brick_wall_001_diffuse_4k.jpg", bitmap_prg_id));
    textures.emplace("alpha_checker", new CheckerTexture(make_float4(1.0), make_float4(0.0f), 20, checker_prg_id));
    textures.emplace("wood", new BitmapTexture("resources/image/plywood_diff_1k.jpg", bitmap_prg_id));

    materials.emplace("black_diffuse", new Diffuse(diffuse_id, textures.at("black")));
    materials.emplace("white_diffuse", new Diffuse(diffuse_id, textures.at("white")));
    materials.emplace("floor", new Diffuse(diffuse_id, textures.at("checker")));
    materials.emplace("brick", new Diffuse(diffuse_id, textures.at("brick")));
    materials.emplace("alpha_checker", new Diffuse(diffuse_id, textures.at("alpha_checker")));
    materials.emplace("orange", new Diffuse(diffuse_id, textures.at("orange")));
    materials.emplace("green", new Diffuse(diffuse_id, textures.at("green")));
    materials.emplace("silver_metal", new Conductor(conductor_id, textures.at("bright_gray")));
    materials.emplace("gold", new Conductor(conductor_id, textures.at("yellow")));
    materials.emplace("glass", new Dielectric(dielectric_id, textures.at("white"), 1.5f));

    auto aniso_disney = make_shared<Disney>(disney_id, textures.at("wine_red"));
    aniso_disney->setRoughness(0.3f);
    aniso_disney->setSubsurface(0.0f);
    aniso_disney->setMetallic(0.9f);
    aniso_disney->setAnisotropic(0.8f);
    materials.emplace("aniso_disney", aniso_disney);

    materials.emplace("image_diffuse", new Diffuse(diffuse_id, textures.at("rtRest")));

    auto wooden_disney = make_shared<Disney>(disney_id, textures.at("wood"));
    wooden_disney->setMetallic(0.9f);
    wooden_disney->setRoughness(0.02f);
    materials.emplace("wooden_disney", wooden_disney);

    lights.emplace("logo1", new AreaEmitter(area_emitter_id, textures.at("orange"), 150.0f, true));
    lights.emplace("logo2", new AreaEmitter(area_emitter_id, textures.at("white"), 300.0f, true));

    shapes.emplace("plane", new Plane(make_float2(-0.5f), make_float2(0.5f)));
    shapes.emplace("sphere", new Sphere(make_float3(0.0f), 1.0f));
    shapes.emplace("cylinder", new Cylinder(0.5f, 1.0f));
    shapes.emplace("box", new Box(make_float3(-0.5f), make_float3(0.5f)));
    auto dragon = make_shared<TriangleMesh>("resources/model/dragon.obj");
    auto bunny = make_shared<TriangleMesh>("resources/model/bunny.obj");
    auto buddha = make_shared<TriangleMesh>("resources/model/happy_vrip_res3.ply");
    auto teapot = make_shared<TriangleMesh>("resources/model/teapot_normal_merged.obj");
    auto armadillo = make_shared<TriangleMesh>("resources/model/Armadillo.ply");

    shared_ptr<TriangleMesh> mitsuba(new TriangleMesh());
    vector<Attributes> mitsuba_mat_attribs;
    mitsuba->loadWithMtl("resources/model/mitsuba-sphere.obj", mitsuba_mat_attribs);

    dragon->calculateNormalSmooth();
    bunny->calculateNormalSmooth();
    buddha->calculateNormalSmooth();
    teapot->calculateNormalSmooth();
    armadillo->calculateNormalSmooth();
    mitsuba->calculateNormalSmooth();
    shapes.emplace("dragon", dragon);
    shapes.emplace("bunny", bunny);
    shapes.emplace("buddha", buddha);
    shapes.emplace("teapot", teapot);
    shapes.emplace("armadillo", armadillo);
    shapes.emplace("mitsuba", mitsuba);

    float x = -logo_aspect * 10 / 2 + 10;

    // Ground plane
    setupObject(plane_prg, shapes.at("plane"), materials.at("floor"), Matrix4f::translate(0, -5, 0) * Matrix4f::scale(500));

    // Dragon
    setupObject(mesh_prg, shapes.at("dragon"), materials.at("glass"),
        Matrix4f::translate(-logo_aspect * 10 / 2 + 10, -0.75, 15) * Matrix4f::rotate(math::pi / 2, { 0, 1, 0 }) * Matrix4f::scale(15));

    // Bunny
    setupObject(mesh_prg, shapes.at("bunny"), materials.at("white_diffuse"), Matrix4f::translate(0, -7.5, 15) * Matrix4f::scale(75));

    // Buddha
    setupObject(mesh_prg, shapes.at("buddha"), materials.at("gold"),
        Matrix4f::translate(logo_aspect * 10 / 2 - 10, -10, 15) * Matrix4f::scale(100));

    // Teapot
    setupObject(mesh_prg, shapes.at("teapot"), materials.at("orange"),
        Matrix4f::translate(-logo_aspect * 10 / 2 + 10, -5, 30) * Matrix4f::scale(3));

    // Armadillo
    setupObject(mesh_prg, shapes.at("armadillo"), materials.at("green"),
        Matrix4f::translate(logo_aspect * 10 / 2 - 10, 0.5, 30) * Matrix4f::rotate(math::pi, { 0, 1, 0 }) * Matrix4f::scale(0.1));

    // Mitsuba
    for (auto& attrib : mitsuba_mat_attribs)
    {
        if (attrib.name == "inside") {
            materials.emplace("inside", new Diffuse(diffuse_id, textures.at("white")));
            addHitgroupRecord(mesh_prg, shapes.at("mitsuba"), materials.at("inside"));
        }
        else if (attrib.name == "case")
        {
            textures.emplace("case", new ConstantTexture(attrib.findOneVec3f("diffuse", Vec3f(0.3, 0.8, 0.7)), constant_prg_id));
            auto case_disney = make_shared<Disney>(disney_id, textures.at("case"));
            case_disney->setRoughness(0.3f);
            case_disney->setSubsurface(0.0f);
            case_disney->setMetallic(0.9f);
            case_disney->setAnisotropic(0.0f);
            materials.emplace("case", case_disney);
            addHitgroupRecord(mesh_prg, shapes.at("mitsuba"), materials.at("case"));
        }
    }
    createGAS(shapes.at("mitsuba"), Matrix4f::translate(0, -4.8, 30) * Matrix4f::scale(5), 2);

    // Sphere1
    setupObject(sphere_prg, shapes.at("sphere"), materials.at("brick"), Matrix4f::translate(x, 0, 0) * Matrix4f::scale(5));

    // Sphere2
    setupObject(sphere_alpha_discard_prg, shapes.at("sphere"), materials.at("black_diffuse"),
        Matrix4f::translate(0, 0, 0) * Matrix4f::scale(5), textures.at("alpha_checker"));

    // Sphere3
    setupObject(sphere_prg, shapes.at("sphere"), materials.at("aniso_disney"), Matrix4f::translate(-x, 0, 0) * Matrix4f::scale(5));

    // Cylinder
    setupObject(cylinder_prg, shapes.at("cylinder"), materials.at("white_diffuse"), Matrix4f::translate(x, -1, -15) * Matrix4f::scale(8));

    // Box
    setupObject(box_prg, shapes.at("box"), materials.at("wooden_disney"),
        Matrix4f::translate(0, -0.9f, -15) * Matrix4f::rotate(math::pi / 6, { 0, 1, 0 }) * Matrix4f::scale(8));

    // Plane
    setupObject(plane_prg, shapes.at("plane"), materials.at("image_diffuse"),
        Matrix4f::translate(-x, 1, -15) * Matrix4f::rotate(math::pi / 2, { 1,0,0 }) * Matrix4f::rotate(-math::pi / 12, { 0, 1, 0 }) * Matrix4f::scale(10));

    setupAreaEmitter(plane_alpha_discard_prg, shapes.at("plane"), lights.at("logo1"),
        Matrix4f::translate(0, 12.5f, -20) * Matrix4f::rotate(math::pi * 2 / 3, { 1.0f, 0.0f, 0.0f }) * Matrix4f::scale(make_float3(logo_aspect * 10, 1, 10)),
        plane_sample_pdf_prg_id, textures.at("logo_alpha"));

    setupAreaEmitter(plane_alpha_discard_prg, shapes.at("plane"), lights.at("logo2"),
        Matrix4f::translate(0, 12.5f, 80) * Matrix4f::rotate(math::pi * 1 / 3, { 1.0f, 0.0f, 0.0f }) * Matrix4f::scale(make_float3(logo_aspect * 10, 1, 10)),
        plane_sample_pdf_prg_id, textures.at("logo_alpha"));

    // Copy light infomation to device
    CUDABuffer<AreaEmitterInfo> d_area_emitter_infos;
    d_area_emitter_infos.copyToDevice(area_emitter_infos);
    params.lights = d_area_emitter_infos.deviceData();
    params.num_lights = static_cast<int>(area_emitter_infos.size());

    CUDA_CHECK(cudaStreamCreate(&stream));
    scene_ias.build(context, stream);
    sbt.createOnDevice();
    params.handle = scene_ias.handle();
    pipeline.create(context);
    d_params.allocate(sizeof(LaunchParams));

#if INTERACTIVE
    // GUI setting
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    const char* glsl_version = "#version 150";
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(pgGetCurrentWindow()->windowPtr(), true);
    ImGui_ImplOpenGL3_Init(glsl_version);
#else 
    float start_time = pgGetElapsedTimef();
    constexpr int num_samples = 100000;
    for (int frame = 0; frame < num_samples; frame += params.samples_per_launch)
    {
        d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

        OPTIX_CHECK(optixLaunch(
            static_cast<OptixPipeline>(pipeline),
            stream,
            d_params.devicePtr(),
            sizeof(LaunchParams),
            &sbt.sbt(),
            params.width,
            params.height,
            1
        ));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_SYNC_CHECK();

        params.frame = frame + 1;
        streamProgress(params.frame, num_samples, pgGetElapsedTimef() - start_time, 20);
    }
    result_bitmap.copyFromDevice();
    result_bitmap.write(pgPathJoin(pgAppDir(), "thumbnail.jpg"));
    pgExit();
#endif
}

// ----------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();

    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

    float start_time = pgGetElapsedTimef();

    optixLaunch(
        static_cast<OptixPipeline>(pipeline),
        stream,
        d_params.devicePtr(),
        sizeof(LaunchParams),
        &sbt.sbt(),
        params.width,
        params.height,
        1
    );

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();
    render_time = pgGetElapsedTimef() - start_time;
    params.frame++;

    result_bitmap.copyFromDevice();
}

// ----------------------------------------------------------------
void App::draw()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Path tracing GUI");

    ImGui::SliderFloat("White", &params.white, 0.01f, 1.0f);
    ImGui::Text("Camera info:");
    ImGui::Text("Origin: %f %f %f", camera.origin().x(), camera.origin().y(), camera.origin().z());
    ImGui::Text("Lookat: %f %f %f", camera.lookat().x(), camera.lookat().y(), camera.lookat().z());
    ImGui::Text("Up: %f %f %f", camera.up().x(), camera.up().y(), camera.up().z());

    float farclip = camera.farClip();
    ImGui::SliderFloat("far clip", &farclip, 500.0f, 10000.0f);
    if (farclip != camera.farClip()) {
        camera.setFarClip(farclip);
        camera_update = true;
    }

    float aperture = camera.aperture();
    ImGui::SliderFloat("Aperture", &aperture, 0.01f, 4.0f);
    if (aperture != camera.aperture()) {
        camera.setAperture(aperture);
        initResultBufferOnDevice();
    }

    float focus_dist = camera.focusDistance();
    ImGui::SliderFloat("Focus distance", &focus_dist, 1.0f, 100.0f);
    if (focus_dist != camera.focusDistance()) {
        camera.setFocusDistance(focus_dist);
        initResultBufferOnDevice();
    }

    auto& light1 = lights.at("logo1");
    float intensity1 = light1->intensity();
    ImGui::SliderFloat("Emittance (light1)", &intensity1, 1.0f, 1000.0f);
    if (intensity1 != light1->intensity())
    {
        light1->setIntensity(intensity1);
        light1->copyToDevice();
        initResultBufferOnDevice();
    }

    auto& light2 = lights.at("logo2");
    float intensity2 = light2->intensity();
    ImGui::SliderFloat("Emittance (light2)", &intensity2, 1.0f, 1000.0f);
    if (intensity2 != light2->intensity())
    {
        light2->setIntensity(intensity2);
        light2->copyToDevice();
        initResultBufferOnDevice();
    }

    ImGui::Text("Frame rate: %.3f ms/frame (%.2f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Render time: %.3f ms/frame", render_time * 1000.0f);
    ImGui::Text("Subframe index: %d", params.frame);

    ImGui::End();

    ImGui::Render();

    result_bitmap.draw(0, 0);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (params.frame == 20000) {
        result_bitmap.write(pgPathJoin(pgAppDir(), "thumbnail.jpg"));
        pgExit();
    }
}

// ----------------------------------------------------------------
void App::close()
{
    env.free();

    for (auto& it : shapes) it.second->free();
    for (auto& it : materials) it.second->free();
    for (auto& it : textures) it.second->free();
    for (auto& it : lights) it.second->free();

    pipeline.destroy();
    context.destroy();

#if INTERACTIVE
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
#endif
}

// ----------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    if (button != MouseButton::Middle) return;
    camera_update = true;
}

// ----------------------------------------------------------------
void App::mouseScrolled(float xoffset, float yoffset)
{
    camera_update = true;
}