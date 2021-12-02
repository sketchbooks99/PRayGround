#include "app.h"

#define INTERACTIVE 0

static void streamProgress(int frame, int max_frame, float elapsed_time, int bar_length)
{
    cout << "\rRendering: [";
    int progress = static_cast<int>( ( (float)(frame) / max_frame) * bar_length );
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
    params.subframe_index = 0;

    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<float4*>(accum_bitmap.devicePtr());

    CUDA_SYNC_CHECK();
}

void App::handleCameraUpdate()
{
    if (!camera_update)
        return;
    camera_update = false;

    float3 U, V, W;
    camera.UVWFrame(U, V, W);

    RaygenRecord* rg_record = reinterpret_cast<RaygenRecord*>(sbt.raygenRecord());
    RaygenData rg_data;
    rg_data.camera =
    {
        .origin = camera.origin(),
        .lookat = camera.lookat(),
        .U = U,
        .V = V,
        .W = W,
        .fov = camera.fov(),
        .aspect = camera.aspect(),
        .aperture = camera.aperture(),
        .focus_distance = camera.focusDistance(),
        .farclip = camera.farClip()
    };

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
    scene_ias = InstanceAccel{InstanceAccel::Type::Instances};

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
    float3 U, V, W;
    camera.UVWFrame(U, V, W);

    // Raygen program
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, raygen_module, "__raygen__lens");
    // Shader binding table data for raygen program
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera =
    {
        .origin = camera.origin(),
        .lookat = camera.lookat(),
        .U = U, 
        .V = V, 
        .W = W,
        .fov = camera.fov(),
        .aspect = camera.aspect(),
        .aperture = camera.aperture(), 
        .focus_distance = camera.focusDistance(),
        .farclip = camera.farClip()
    };
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
    uint32_t constant_prg_id = setupCallable(textures_module, DC_FUNC_STR("constant"), "");
    uint32_t checker_prg_id = setupCallable(textures_module, DC_FUNC_STR("checker"), "");
    uint32_t bitmap_prg_id = setupCallable(textures_module, DC_FUNC_STR("bitmap"), "");

    // Callable programs for surfaces 
    // Diffuse
    uint32_t diffuse_sample_bsdf_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("sample_diffuse"), CC_FUNC_STR("bsdf_diffuse"));
    uint32_t diffuse_pdf_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("pdf_diffuse"), "");
    // Conductor
    uint32_t conductor_sample_bsdf_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("sample_conductor"), CC_FUNC_STR("bsdf_conductor"));
    uint32_t conductor_pdf_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("pdf_conductor"), "");
    // Dielectric
    uint32_t dielectric_sample_bsdf_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("sample_dielectric"), CC_FUNC_STR("bsdf_dielectric"));
    uint32_t dielectric_pdf_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("pdf_dielectric"), "");
    // Disney
    uint32_t disney_sample_bsdf_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("sample_disney"), CC_FUNC_STR("bsdf_disney"));
    uint32_t disney_pdf_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("pdf_disney"), "");
    // AreaEmitter
    uint32_t area_emitter_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("area_emitter"), "");
    
    // Callable program for direct sampling of area emitter
    uint32_t plane_sample_pdf_prg_id = setupCallable(hitgroups_module, DC_FUNC_STR("rnd_sample_plane"), CC_FUNC_STR("pdf_plane"));

    textures.emplace("env", new FloatBitmapTexture("resources/image/christmas_photo_studio_01_4k.exr", bitmap_prg_id));

    env = EnvironmentEmitter{textures.at("env")};
    env.copyToDevice();

    // Miss program
    ProgramGroup miss_prg = pipeline.createMissProgram(context, miss_module, MS_FUNC_STR("envmap"));
    // Shader binding table data for miss program
    MissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env.devicePtr();
    sbt.setMissRecord(miss_record);

    // Hitgroup program
    // Plane
    auto plane_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("plane"), IS_FUNC_STR("plane"));
    auto plane_alpha_discard_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("plane"), IS_FUNC_STR("plane"), AH_FUNC_STR("alpha_discard"));
    // Sphere
    auto sphere_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("sphere"), IS_FUNC_STR("sphere"));
    auto sphere_alpha_discard_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("sphere"), IS_FUNC_STR("sphere"), AH_FUNC_STR("alpha_discard"));
    // Box
    auto box_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("box"), IS_FUNC_STR("box"));
    // Cylinder
    auto cylinder_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("cylinder"), IS_FUNC_STR("cylinder"));
    // Triangle mesh
    auto mesh_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("mesh"));

    struct Primitive
    {
        shared_ptr<Shape> shape;
        shared_ptr<Material> material;
        uint32_t sample_bsdf_id;
        uint32_t pdf_id;
    };

    uint32_t sbt_idx = 0;
    uint32_t sbt_offset = 0;
    uint32_t instance_id = 0;

    using SurfaceP = variant<shared_ptr<Material>, shared_ptr<AreaEmitter>>;
    auto addHitgroupRecord = [&](ProgramGroup& prg, shared_ptr<Shape> shape, SurfaceP surface, uint32_t sample_bsdf_id, uint32_t pdf_id, shared_ptr<Texture> alpha_texture = nullptr)
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
                .sample_id = sample_bsdf_id,
                .bsdf_id = sample_bsdf_id,
                .pdf_id = pdf_id,
                .type = is_mat ? std::get<shared_ptr<Material>>(surface)->surfaceType() : SurfaceType::AreaEmitter,
            },
            .alpha_texture =
            {
                alpha_texture ? alpha_texture->devicePtr() : nullptr,
                alpha_texture ? alpha_texture->programId() : bitmap_prg_id
            }
        };

        sbt.addHitgroupRecord(record);
        sbt_idx++;
    };

    auto createGAS = [&](shared_ptr<Shape> shape, const Matrix4f& transform, uint32_t num_sbt=1)
    {
        // Build GAS and add it to IAS
        ShapeInstance instance{shape->type(), shape, transform};
        instance.allowCompaction();
        instance.buildAccel(context, stream);
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);

        scene_ias.addInstance(instance);

        instance_id++;
        sbt_offset += ThumbnailSBT::NRay * num_sbt;
    };

    auto setupPrimitive = [&](ProgramGroup& prg, const Primitive& p, const Matrix4f& transform, shared_ptr<Texture> alpha_texture = nullptr)
    {
        addHitgroupRecord(prg, p.shape, p.material, p.sample_bsdf_id, p.pdf_id, alpha_texture);
        createGAS(p.shape, transform);
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

        addHitgroupRecord(prg, shape, area, area_emitter_prg_id, area_emitter_prg_id, alpha_texture);
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

    materials.emplace("black_diffuse", new Diffuse(textures.at("black")));
    materials.emplace("white_diffuse", new Diffuse(textures.at("white")));
    materials.emplace("floor", new Diffuse(textures.at("checker")));
    materials.emplace("brick", new Diffuse(textures.at("brick")));
    materials.emplace("alpha_checker", new Diffuse(textures.at("alpha_checker")));
    materials.emplace("orange", new Diffuse(textures.at("orange")));
    materials.emplace("green", new Diffuse(textures.at("green")));
    materials.emplace("silver_metal", new Conductor(textures.at("bright_gray")));
    materials.emplace("gold", new Conductor(textures.at("yellow")));
    materials.emplace("glass", new Dielectric(textures.at("white"), 1.5f));

    auto aniso_disney = make_shared<Disney>(textures.at("wine_red"));
    aniso_disney->setRoughness(0.3f);
    aniso_disney->setSubsurface(0.0f);
    aniso_disney->setMetallic(0.9f);
    aniso_disney->setAnisotropic(0.8f);
    materials.emplace("aniso_disney", aniso_disney);

    materials.emplace("image_diffuse", new Diffuse(textures.at("rtRest")));

    auto wooden_disney = make_shared<Disney>(textures.at("wood"));
    wooden_disney->setMetallic(0.9f);
    wooden_disney->setRoughness(0.02f);
    materials.emplace("wooden_disney", wooden_disney);
    
    lights.emplace("logo1", new AreaEmitter( textures.at("orange"), 150.0f, true ));
    lights.emplace("logo2", new AreaEmitter( textures.at("white"),  300.0f, true ));

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
    
    dragon->smooth();
    bunny->smooth();
    buddha->smooth();
    teapot->smooth();
    armadillo->smooth();
    mitsuba->smooth();
    shapes.emplace("dragon", dragon);
    shapes.emplace("bunny", bunny);
    shapes.emplace("buddha", buddha);
    shapes.emplace("teapot", teapot);
    shapes.emplace("armadillo", armadillo);
    shapes.emplace("mitsuba", mitsuba);

    float x = -logo_aspect * 10 / 2 + 10;

    // Ground plane
    Primitive p1{ shapes.at("plane"), materials.at("floor"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(plane_prg, p1, Matrix4f::translate(0, -5, 0) * Matrix4f::scale(500));

    // Dragon
    Primitive p3{ shapes.at("dragon"), materials.at("glass"), dielectric_sample_bsdf_prg_id, dielectric_pdf_prg_id };
    setupPrimitive(mesh_prg, p3, Matrix4f::translate(-logo_aspect * 10 / 2 + 10, -0.75, 15) *Matrix4f::rotate(math::pi/2, {0, 1, 0})* Matrix4f::scale(15));
    
    // Bunny
    Primitive p4{ shapes.at("bunny"), materials.at("white_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(mesh_prg, p4, Matrix4f::translate(0, -7.5, 15)* Matrix4f::scale(75));

    // Buddha
    Primitive p5{ shapes.at("buddha"), materials.at("gold"), conductor_sample_bsdf_prg_id, conductor_pdf_prg_id };
    setupPrimitive(mesh_prg, p5, Matrix4f::translate(logo_aspect * 10 / 2 - 10, -10, 15)* Matrix4f::scale(100));

    // Teapot
    Primitive p6{ shapes.at("teapot"), materials.at("orange"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(mesh_prg, p6, Matrix4f::translate(-logo_aspect * 10 / 2 + 10, -5, 30)* Matrix4f::scale(3));

    // Armadillo
    Primitive p7{ shapes.at("armadillo"), materials.at("green"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(mesh_prg, p7, Matrix4f::translate(logo_aspect * 10 / 2 - 10, 0.5, 30)* Matrix4f::rotate(math::pi, { 0, 1, 0 }) * Matrix4f::scale(0.1));

    // Mitsuba
    for (auto& attrib : mitsuba_mat_attribs)
    {
        if (attrib.name == "inside") {
            materials.emplace("inside", new Diffuse(textures.at("white")));
            addHitgroupRecord(mesh_prg, shapes.at("mitsuba"), materials.at("inside"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id);
        }
        else if (attrib.name == "case")
        {
            textures.emplace("case", new ConstantTexture(attrib.findOneFloat3("diffuse", make_float3(0.3, 0.8, 0.7)), constant_prg_id));
            auto case_disney = make_shared<Disney>(textures.at("case"));
            case_disney->setRoughness(0.3f);
            case_disney->setSubsurface(0.0f);
            case_disney->setMetallic(0.9f);
            case_disney->setAnisotropic(0.8f);
            materials.emplace("case", case_disney);
            addHitgroupRecord(mesh_prg, shapes.at("mitsuba"), materials.at("case"), disney_sample_bsdf_prg_id, disney_pdf_prg_id);
        }
    }
    createGAS(shapes.at("mitsuba"), Matrix4f::translate(0, -4.8, 30) * Matrix4f::scale(5), 2);

    // Sphere1
    Primitive p9{ shapes.at("sphere"), materials.at("brick"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(sphere_prg, p9, Matrix4f::translate(x, 0, 0)* Matrix4f::scale(5));
    
    // Sphere2
    Primitive p10{ shapes.at("sphere"), materials.at("black_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(sphere_alpha_discard_prg, p10, Matrix4f::translate(0, 0, 0)* Matrix4f::scale(5), textures.at("alpha_checker"));

    // Sphere3
    Primitive p11{ shapes.at("sphere"), materials.at("aniso_disney"), disney_sample_bsdf_prg_id, disney_pdf_prg_id };
    setupPrimitive(sphere_prg, p11, Matrix4f::translate(-x, 0, 0)* Matrix4f::scale(5));

    // Cylinder
    Primitive p12{ shapes.at("cylinder"), materials.at("white_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(cylinder_prg, p12, Matrix4f::translate(x, -1, -15)* Matrix4f::scale(8));

    // Box
    Primitive p13{ shapes.at("box"), materials.at("wooden_disney"), disney_sample_bsdf_prg_id, disney_pdf_prg_id };
    setupPrimitive(box_prg, p13, Matrix4f::translate(0, -0.9f, -15) * Matrix4f::rotate(math::pi / 6, {0, 1, 0}) * Matrix4f::scale(8));

    // Plane
    Primitive p14{ shapes.at("plane"), materials.at("image_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id};
    setupPrimitive(plane_prg, p14, Matrix4f::translate(-x, 1, -15) * Matrix4f::rotate(math::pi / 2, { 1,0,0 }) * Matrix4f::rotate(-math::pi / 12, { 0, 1, 0 }) * Matrix4f::scale(10));

    setupAreaEmitter(plane_alpha_discard_prg, shapes.at("plane"), lights.at("logo1"), 
        Matrix4f::translate(0, 12.5f, -20) * Matrix4f::rotate(math::pi * 2 / 3, { 1.0f, 0.0f, 0.0f })* Matrix4f::scale(make_float3(logo_aspect * 10, 1, 10)), 
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

        params.subframe_index = frame + 1;
        streamProgress(params.subframe_index, num_samples, pgGetElapsedTimef() - start_time, 20);
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
    params.subframe_index++;

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
    ImGui::Text("Origin: %f %f %f", camera.origin().x, camera.origin().y, camera.origin().z);
    ImGui::Text("Lookat: %f %f %f", camera.lookat().x, camera.lookat().y, camera.lookat().z);
    ImGui::Text("Up: %f %f %f", camera.up().x, camera.up().y, camera.up().z);

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
    ImGui::Text("Subframe index: %d", params.subframe_index);

    ImGui::End();

    ImGui::Render();

    result_bitmap.draw(0, 0);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (params.subframe_index == 20000) {
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
