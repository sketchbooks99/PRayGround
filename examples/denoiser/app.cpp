#include "app.h"

// --------------------------------------------------------------------
void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocateDevicePtr();
    normal_bitmap.allocateDevicePtr();
    albedo_bitmap.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<float4*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<float4*>(accum_bitmap.devicePtr());
    params.normal_buffer = reinterpret_cast<float4*>(normal_bitmap.devicePtr());
    params.albedo_buffer = reinterpret_cast<float4*>(albedo_bitmap.devicePtr());

    CUDA_SYNC_CHECK();
}

// --------------------------------------------------------------------
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
        .W = W
    };
}

// --------------------------------------------------------------------
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
    ias = InstanceAccel{InstanceAccel::Type::Instances};

    // Pipeline settings
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(5);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);

    // Create modules from cuda source file 
    Module raygen_module, miss_module, hitgroups_module, textures_module, surfaces_module;
    raygen_module = pipeline.createModuleFromCudaFile(context, "cuda/raygen.cu");
    miss_module = pipeline.createModuleFromCudaFile(context, "cuda/miss.cu");
    hitgroups_module = pipeline.createModuleFromCudaFile(context, "cuda/hitgroups.cu");
    textures_module = pipeline.createModuleFromCudaFile(context, "cuda/textures.cu");
    surfaces_module = pipeline.createModuleFromCudaFile(context, "cuda/surfaces.cu");

    // Initialize bitmaps to store rendered results
    const int32_t width = pgGetWidth();
    const int32_t height = pgGetHeight();
    result_bitmap.allocate(PixelFormat::RGBA, width, height);
    accum_bitmap.allocate(PixelFormat::RGBA, width, height);
    normal_bitmap.allocate(PixelFormat::RGBA, width, height);
    albedo_bitmap.allocate(PixelFormat::RGBA, width, height);
    initResultBufferOnDevice();

    // Configuration of launch parameters
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.samples_per_launch = 1;
    params.max_depth = 5;

    // Camera settings
    camera.setOrigin(278.0f, 273.0f, -900.0f);
    camera.setLookat(278.0f, 273.0f, 330.0f);
    camera.setUp(0, 1, 0);
    camera.setFov(35.0f);
    camera.setAspect((float)width / height);
    camera.enableTracking(pgGetCurrentWindow());

    float3 U, V, W;
    camera.UVWFrame(U, V, W);

    // Raygen program
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, raygen_module, "__raygen__pinhole");
    // Shader binding table data for raygen program
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera = 
    {
        .origin = camera.origin(), 
        .lookat = camera.lookat(), 
        .U = U, 
        .V = V, 
        .W = W
    };
    sbt.setRaygenRecord(raygen_record);

    auto setupCallable = [&](const Module& module, const string& dc, const string& cc) 
        -> uint32_t
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

    auto black = make_shared<ConstantTexture>(make_float3(0.0f), constant_prg_id);
    env = EnvironmentEmitter{black};
    env.copyToDevice();

    // Miss program
    ProgramGroup miss_prg = pipeline.createMissProgram(context, miss_module, MS_FUNC_STR("envmap"));
    // Shader binding table data for miss program
    MissRecord miss_record = {};
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env.devicePtr();
    sbt.setMissRecord(miss_record);

    // Hitgroupプログラム
    // Plane
    auto plane_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("plane"), IS_FUNC_STR("plane"));
    // Sphere
    auto sphere_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("sphere"), IS_FUNC_STR("sphere"));
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
    
    auto setupPrimitive = [&](ProgramGroup& prg, const Primitive& primitive, const Matrix4f& transform)
    {
        // Copy data to GPU
        primitive.shape->copyToDevice();
        primitive.shape->setSbtIndex(sbt_idx);
        primitive.material->copyToDevice();

        // Register data to shader binding table
        HitgroupRecord record;
        prg.recordPackHeader(&record);
        record.data = 
        {
            .shape_data = primitive.shape->devicePtr(), 
            .surface_info = 
            {
                .data = primitive.material->devicePtr(),
                .sample_id = primitive.sample_bsdf_id,
                .bsdf_id = primitive.sample_bsdf_id,
                .pdf_id = primitive.pdf_id,
                .type = primitive.material->surfaceType()
            }
        };

        sbt.addHitgroupRecord(record);
        sbt_idx++;

        // Build GAS and add it to IAS
        ShapeInstance instance{primitive.shape->type(), primitive.shape, transform};
        instance.allowCompaction();
        instance.buildAccel(context, stream);
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);

        ias.addInstance(instance);

        instance_id++;
        sbt_offset += DenoiserSBT::NRay;
    };

    auto setupAreaEmitter = [&](
        ProgramGroup& prg, 
        shared_ptr<Shape> shape,
        AreaEmitter area, Matrix4f transform, 
        uint32_t sample_pdf_id
    )
    {        
        ASSERT(dynamic_pointer_cast<Plane>(shape) || dynamic_pointer_cast<Sphere>(shape), "The shape of area emitter must be a plane or sphere.");

        shape->copyToDevice();
        shape->setSbtIndex(sbt_idx);
        area.copyToDevice();

        HitgroupRecord record;
        prg.recordPackHeader(&record);
        record.data = 
        {
            .shape_data = shape->devicePtr(), 
            .surface_info = 
            {
                .data = area.devicePtr(),
                .sample_id = area_emitter_prg_id, // not used
                .bsdf_id = area_emitter_prg_id,
                .pdf_id = area_emitter_prg_id,    // not used
                .type = SurfaceType::AreaEmitter
            }
        };

        sbt_idx++;

        sbt.addHitgroupRecord(record);

        // Build GAS and add it to IAS
        ShapeInstance instance{shape->type(), shape, transform};
        instance.allowCompaction();
        instance.buildAccel(context, stream);
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);

        ias.addInstance(instance);

        instance_id++;
        sbt_offset += DenoiserSBT::NRay;
    };

    // Scene ----------------------------------------------------------
    // Shapes
    auto bunny = make_shared<TriangleMesh>("resources/model/bunny.obj");
    bunny->smooth();
    shapes.emplace("bunny", bunny);
    shapes.emplace("wall", make_shared<Plane>(make_float2(0, 0), make_float2(556, 548)));
    shapes.emplace("ceiling_light", make_shared<Plane>(make_float2(213, 227), make_float2(343, 332)));

    // Textures
    textures.emplace("green", make_shared<ConstantTexture>(make_float3(0.05, 0.8, 0.05), constant_prg_id));
    textures.emplace("red", make_shared<ConstantTexture>(make_float3(0.8, 0.05, 0.05), constant_prg_id));
    textures.emplace("wall_white", make_shared<ConstantTexture>(make_float3(0.8), constant_prg_id));
    textures.emplace("white", make_shared<ConstantTexture>(make_float3(1.0f), constant_prg_id));
    textures.emplace("checker", make_shared<CheckerTexture>(make_float3(0.9f), make_float3(0.3f), checker_prg_id));
    textures.emplace("black", black);
    textures.emplace("gray", make_shared<ConstantTexture>(make_float3(0.25), constant_prg_id));

    // Materials
    materials.emplace("green_diffuse", make_shared<Diffuse>(textures.at("green")));
    materials.emplace("red_diffuse", make_shared<Diffuse>(textures.at("red")));
    materials.emplace("wall_diffuse", make_shared<Diffuse>(textures.at("wall_white")));
    auto disney = make_shared<Disney>(textures.at("gray"));
    disney->setRoughness(0.5f);
    disney->setMetallic(0.9f);
    materials.emplace("gray_disney", disney);

    // Area emitter
    AreaEmitter area{textures.at("white"), 15.0f, true};

    // Bunny
    Primitive bunnyP(shapes.at("bunny"), materials.at("gray_disney"), disney_sample_bsdf_prg_id, disney_pdf_prg_id);
    setupPrimitive(mesh_prg, bunnyP, Matrix4f::translate(278, 273, 0) * Matrix4f::scale(500.0f));

    // Left wall 
    Primitive l_wall(shapes.at("wall"), materials.at("green_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id);
    setupPrimitive(plane_prg, l_wall, Matrix4f::translate(0, 273, 0) * Matrix4f::rotate(math::pi / 2, {0.0, 0.0f, 1.0}));

    // Right wall 
    Primitive r_wall(shapes.at("wall"), materials.at("red_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id);
    setupPrimitive(plane_prg, r_wall, Matrix4f::translate(556, 273, 0) * Matrix4f::rotate(math::pi / 2, {0.0, 0.0f, 1.0}));

    // Back wall
    Primitive b_wall(shapes.at("wall"), materials.at("wall_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id);
    setupPrimitive(plane_prg, b_wall, Matrix4f::translate(0, 273, 559) * Matrix4f::rotate(math::pi / 2, {1.0, 0.0f, 0.0}));

    // Ceiling 
    Primitive ceiling(shapes.at("wall"), materials.at("wall_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id);
    setupPrimitive(plane_prg, ceiling, Matrix4f::translate(278, 548, 279.5));

    // Floor 
    Primitive floor(shapes.at("wall"), materials.at("checker"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id);
    setupPrimitive(plane_prg, floor, Matrix4f::translate(278, 0, 279.5));

    // Ceiling light
    setupAreaEmitter(plane_prg, shapes.at("ceiling_light"), area, Matrix4f::translate(278, 547.9, 279.5), area_emitter_prg_id);

    // Denoiser settings
    denoise_data.width = result_bitmap.width();
    denoise_data.height = result_bitmap.height();
    denoise_data.outputs.push_back(new float[denoise_data.width * denoise_data.height * 4]);

    // Prepare rendering
    CUDA_CHECK(cudaStreamCreate(&stream));
    ias.build(context, stream);
    sbt.createOnDevice();
    params.handle = ias.handle();
    pipeline.create(context);
    d_params.allocate(sizeof(LaunchParams));

    // GUI setting
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    const char* glsl_version = "#version 330";
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(pgGetCurrentWindow()->windowPtr(), true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

// --------------------------------------------------------------------
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

    // Fetch rendered result from GPU
    result_bitmap.copyFromDevice();
    normal_bitmap.copyFromDevice();
    albedo_bitmap.copyFromDevice();

    denoise_data.color = reinterpret_cast<float*>(result_bitmap.devicePtr());
    denoise_data.albedo = reinterpret_cast<float*>(albedo_bitmap.devicePtr());
    denoise_data.normal = reinterpret_cast<float*>(normal_bitmap.devicePtr());

    if (params.frame == 0)
        denoiser.create(context, denoise_data, 0, 0, false, false);
    else
        denoiser.update(denoise_data);

    denoiser.run();

    denoiser.copyFromDevice();
}

// --------------------------------------------------------------------
void App::draw()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Denoiser");

    ImGui::End();

    ImGui::Render();

    denoiser.draw(denoise_data);
}

void App::close()
{
    denoiser.destroy();
}

// --------------------------------------------------------------------
void App::keyPressed(int key)
{

}

// --------------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    camera_update = true;
}

void App::mouseScrolled(float x, float y)
{
    camera_update = true;
}

