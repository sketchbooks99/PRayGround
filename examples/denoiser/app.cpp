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
    rg_data.camera = camera.getData();
    CUDA_CHECK(cudaMemcpy(&rg_record->data, &rg_data, sizeof(RaygenData), cudaMemcpyHostToDevice));

    initResultBufferOnDevice();
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
    pipeline.setNumAttributes(6);

    // Create modules from cuda source file 
    Module raygen_module, miss_module, hitgroups_module, textures_module, surfaces_module;
    raygen_module = pipeline.createModuleFromCudaFile(context, "cuda/raygen.cu");
    miss_module = pipeline.createModuleFromCudaFile(context, "cuda/miss.cu");
    hitgroups_module = pipeline.createModuleFromCudaFile(context, "cuda/hitgroups.cu");
    textures_module = pipeline.createModuleFromCudaFile(context, "cuda/textures.cu");
    surfaces_module = pipeline.createModuleFromCudaFile(context, "cuda/surfaces.cu");

    // Initialize bitmaps to store rendered results
    const int32_t width = pgGetWidth() / 2;
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
    params.white = 1.0f;

    // Camera settings
    camera.setOrigin(275, 275, -900);
    camera.setLookat(275, 275, 330);
    camera.setUp(0, 1, 0);
    camera.setFov(35.0f);
    camera.setAspect((float)width / height);
    camera.setFarClip(5000);
    camera.enableTracking(pgGetCurrentWindow());

    // Raygen program
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, raygen_module, "__raygen__pinhole");
    // Shader binding table data for raygen program
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera = camera.getData();
    sbt.setRaygenRecord(raygen_record);

    auto setupCallable = [&](const Module& module, const string& dc, const string& cc) -> uint32_t
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

    // Hitgroup program
    // Plane
    auto plane_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("plane"), IS_FUNC_STR("plane"));
    // Sphere
    auto sphere_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("sphere"), IS_FUNC_STR("sphere"));
    // Triangle mesh
    auto mesh_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("mesh"));

    // Callable program for direct sampling of area emitter
    uint32_t plane_sample_pdf_prg_id = setupCallable(hitgroups_module, DC_FUNC_STR("rnd_sample_plane"), CC_FUNC_STR("pdf_plane"));

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

    vector<AreaEmitterInfo> area_emitter_infos;
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

        AreaEmitterInfo infos =
        {
            .shape_data = shape->devicePtr(), 
            .objToWorld = transform,
            .worldToObj = transform.inverse(),
            .sample_id = sample_pdf_id, 
            .pdf_id = sample_pdf_id
        };
        area_emitter_infos.emplace_back(infos);
    };

    // Scene ----------------------------------------------------------
    // Shapes
    auto bunny = new TriangleMesh("resources/model/bunny.obj");
    bunny->smooth();
    shapes.emplace("bunny", bunny);
    shapes.emplace("wall", new Plane(make_float2(-275, -275), make_float2(275, 275)));
    shapes.emplace("ceiling_light", new Plane(make_float2(-60, -60), make_float2(60, 60)));
    shapes.emplace("sphere", new Sphere(make_float3(0.0f), 1.0f));

    // Textures
    textures.emplace("green", new ConstantTexture(make_float3(0.05, 0.8, 0.05), constant_prg_id));
    textures.emplace("red", new ConstantTexture(make_float3(0.8, 0.05, 0.05), constant_prg_id));
    textures.emplace("wall_white", new ConstantTexture(make_float3(0.8), constant_prg_id));
    textures.emplace("white", new ConstantTexture(make_float3(1.0f), constant_prg_id));
    textures.emplace("checker", new CheckerTexture(make_float3(0.9f), make_float3(0.3f), 10, checker_prg_id));
    textures.emplace("black", black);
    textures.emplace("gray", new ConstantTexture(make_float3(0.25), constant_prg_id));
    textures.emplace("orange", new ConstantTexture(make_float3(0.8, 0.7, 0.3), constant_prg_id));

    // Materials
    materials.emplace("green_diffuse", new Diffuse(textures.at("green")));
    materials.emplace("red_diffuse", new Diffuse(textures.at("red")));
    materials.emplace("wall_diffuse", new Diffuse(textures.at("wall_white")));
    materials.emplace("floor_diffuse", new Diffuse(textures.at("checker")));
    materials.emplace("glass", new Dielectric(textures.at("white"), 1.5f));
    materials.emplace("metal", new Conductor(textures.at("orange")));
    auto disney = new Disney(textures.at("gray"));
    disney->setRoughness(0.5f);
    disney->setMetallic(0.9f);
    materials.emplace("gray_disney", disney);

    // Area emitter
    AreaEmitter area{textures.at("white"), 15.0f, true};

    // Bunny
    Primitive bunnyP{ shapes.at("bunny"), materials.at("gray_disney"), disney_sample_bsdf_prg_id, disney_pdf_prg_id };
    setupPrimitive(mesh_prg, bunnyP, Matrix4f::translate(275, 150, 275)* Matrix4f::rotate(math::pi, {0.0f, 1.0f, 0.0f}) * Matrix4f::scale(1200.0f));

    // Left wall 
    Primitive l_wall{ shapes.at("wall"), materials.at("green_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(plane_prg, l_wall, Matrix4f::translate(550, 275, 275) * Matrix4f::rotate(math::pi / 2, {0.0, 0.0f, 1.0}));

    // Right wall 
    Primitive r_wall{ shapes.at("wall"), materials.at("red_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(plane_prg, r_wall, Matrix4f::translate(0, 275, 275) * Matrix4f::rotate(math::pi / 2, {0.0, 0.0f, 1.0}));

    // Back wall
    Primitive b_wall{ shapes.at("wall"), materials.at("wall_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(plane_prg, b_wall, Matrix4f::translate(275, 275, 550) * Matrix4f::rotate(math::pi / 2, {1.0, 0.0f, 0.0}));

    // Ceiling 
    Primitive ceiling{ shapes.at("wall"), materials.at("wall_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(plane_prg, ceiling, Matrix4f::translate(275, 550, 275));

    // Floor 
    Primitive floor{ shapes.at("wall"), materials.at("floor_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(plane_prg, floor, Matrix4f::translate(275, 0, 275));

    // Glass sphere
    Primitive glass_sphere{ shapes.at("sphere"), materials.at("glass"), dielectric_sample_bsdf_prg_id, dielectric_pdf_prg_id };
    setupPrimitive(sphere_prg, glass_sphere, Matrix4f::translate(150, 100, 150) * Matrix4f::scale(100));

    // Metal sphere
    Primitive metal_sphere{ shapes.at("sphere"), materials.at("metal"), conductor_sample_bsdf_prg_id, conductor_pdf_prg_id };
    setupPrimitive(sphere_prg, metal_sphere, Matrix4f::translate(375, 120, 350)* Matrix4f::scale(120));

    // Ceiling light
    setupAreaEmitter(plane_prg, shapes.at("ceiling_light"), area, Matrix4f::translate(275, 545, 275), plane_sample_pdf_prg_id);

    // Setup area emitter on device
    CUDABuffer<AreaEmitterInfo> d_area_emitter_infos;
    d_area_emitter_infos.copyToDevice(area_emitter_infos);
    params.lights = d_area_emitter_infos.deviceData();
    params.num_lights = static_cast<uint32_t>(area_emitter_infos.size());

    // Denoiser settings
    denoise_data.width = result_bitmap.width();
    denoise_data.height = result_bitmap.height();
    denoise_data.outputs.push_back(new float[denoise_data.width * denoise_data.height * 4]);
    denoise_data.color = result_bitmap.devicePtr();
    denoise_data.albedo = albedo_bitmap.devicePtr();
    denoise_data.normal = normal_bitmap.devicePtr();
    denoiser.init(context, denoise_data, 0, 0, false, false);

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

    // Fetch rendered result from GPU
    result_bitmap.copyFromDevice();
    normal_bitmap.copyFromDevice();
    albedo_bitmap.copyFromDevice();

    denoise_data.color = result_bitmap.devicePtr();
    denoise_data.albedo = albedo_bitmap.devicePtr();
    denoise_data.normal = normal_bitmap.devicePtr();

    denoiser.update(denoise_data);

    start_time = pgGetElapsedTimef();
    denoiser.run();
    denoise_time = pgGetElapsedTimef() - start_time;

    denoiser.copyFromDevice();

    params.frame++;
}

// --------------------------------------------------------------------
void App::draw()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    const float start_time = pgGetElapsedTimef();
    result_bitmap.draw();
    denoiser.draw(denoise_data, pgGetWidth() / 2, 0);
    const float display_time = pgGetElapsedTimef() - start_time;

    ImGui::Begin("Denoiser");

    ImGui::Text("Camera info:");
    ImGui::Text("Origin: %f %f %f", camera.origin().x, camera.origin().y, camera.origin().z);
    ImGui::Text("Lookat: %f %f %f", camera.lookat().x, camera.lookat().y, camera.lookat().z);

    ImGui::Text("Frame rate: %.3f s/frame (%.2f FPS)", ImGui::GetIO().DeltaTime, ImGui::GetIO().Framerate);
    ImGui::Text("Frame: %d", params.frame);
    ImGui::Text("Render time: %.3f s/frame", render_time);
    ImGui::Text("Denoise time: %.3f s/frame", denoise_time);
    ImGui::Text("Display time: %.3f s/frame", display_time);

    ImGui::End();

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void App::close()
{
    //denoiser.destroy();
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

