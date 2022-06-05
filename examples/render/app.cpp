#include "app.h"

// ------------------------------------------------------------------
void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.devicePtr());

    CUDA_SYNC_CHECK();
}

// ------------------------------------------------------------------
void App::handleCameraUpdate()
{
    if (!camera_update)
        return;
    camera_update = false;

    pgRaygenRecord<Camera>* rg_record = reinterpret_cast<pgRaygenRecord<Camera>*>(sbt.raygenRecord());
    pgRaygenData<Camera> rg_data;
    rg_data.camera = camera.getData();

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(&rg_record->data), 
        &rg_data, sizeof(pgRaygenData<Camera>), 
        cudaMemcpyHostToDevice
    ));

    initResultBufferOnDevice();
}

// ------------------------------------------------------------------
void App::launchGenLightVertices()
{

    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

    optixLaunch(
        static_cast<OptixPipeline>(genlight_pipeline),
        stream,
        d_params.devicePtr(),
        sizeof(LaunchParams),
        &sbt.sbt(),
        params.width,
        params.height,
        1
    );
}

// ------------------------------------------------------------------
// Initalization of VCM for light vertices generation
// ------------------------------------------------------------------
void App::initVCMIteration()
{
    VCM& vcm = params.vcm;
    const int path_count = params.width * params.height;

    const float light_subpath_count = float(path_count);

    // Setup radius, 1st iteration has vcm.iteraction == 0, thus offset
    float radius = vcm.base_radius;
    radius /= powf(float(vcm.iteration + 1), 0.5f + (1.0f - vcm.radius_alpha));
    radius = std::max(radius, 1e-7f);
    vcm.radius = radius;
    const float radius2 = pow2(radius);
    
    // Factor used to normalize vertex merging contribution
    vcm.vm_normalization = 1.0f / (radius2 * math::pi * light_subpath_count);

    const float eta_vcm = (math::pi * radius2) * light_subpath_count;
    vcm.mis_vm_weight_factor = eta_vcm;
    vcm.mis_vc_weight_factor = 1.0f / eta_vcm;

    // Fill path_ends with zero.
    vcm.path_ends.resize(path_count);
    thrust::fill(vcm.path_ends.begin(), vcm.path_ends.end(), 0);

    vcm.light_vertices.reserve(path_count);
    vcm.light_vertices.clear();
}

// ------------------------------------------------------------------
void App::setup()
{
    // Initialize CUDA
    stream = 0; 
    CUDA_CHECK(cudaFree(0));

    // Initialize OptixDeviceContext
    OPTIX_CHECK(optixInit());
    context.disableValidation();
    context.create();

    // Initialize instance accel structure
    ias = InstanceAccel{ InstanceAccel::Type::Instances };

    // Pipeline settings
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(5);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(2);
    
    genlight_pipeline.setCompileOptions(pipeline.compileOptions());
    genlight_pipeline.setLinkOptions(pipeline.linkOptions());

    // Module creation
    Module rg_module, ms_module, is_module, ch_module, tex_module, surf_module, light_sample_module;
    rg_module = pipeline.createModuleFromCudaFile(context, "cuda/raygen.cu");
    ms_module = pipeline.createModuleFromCudaFile(context, "cuda/miss.cu");
    is_module = pipeline.createModuleFromCudaFile(context, "prayground/optix/cuda/intersection.cu");
    ch_module = pipeline.createModuleFromCudaFile(context, "cuda/closesthit.cu");
    tex_module = pipeline.createModuleFromCudaFile(context, "cuda/textures.cu");
    surf_module = pipeline.createModuleFromCudaFile(context, "cuda/surfaces.cu");
    light_sample_module = pipeline.createModuleFromCudaFile(context, "cuda/light_sample.cu");

    // Initialize bitmap
    result_bmp.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());
    accum_bmp.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());

    // Configuration of launch parameters
    params.width = result_bmp.width();
    params.height = result_bmp.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;

    // VCM settings
    VCM vcm;

    initResultBufferOnDevice();

    // Camera settings
    camera.setOrigin(0, 0, -100);
    camera.setLookat(0, 0, 0);
    camera.setUp(0, 1, 0);
    camera.setFarClip(5000);
    camera.setAspect(40.0f);
    camera.enableTracking(pgGetCurrentWindow());
    
    ProgramGroup rg_genlight_prg;
    rg_genlight_prg.createRaygen(context, rg_module, "__raygen__lightpath");
    pgRaygenRecord<Camera> rg_record;
    rg_genlight_prg.recordPackHeader(&rg_record);
    sbt.setRaygenRecord(rg_record);

    // Raygen program for camera path
    ProgramGroup rg_prg = pipeline.createRaygenProgram(context, rg_module, "__raygen__camerapath");

    auto setupCallable = [&](const Module& m, const string& dc, const string& cc) -> uint32_t
    {
        pgCallableRecord ca_record{};
        auto [prg, id] = pipeline.createCallablesProgram(context, m, dc, cc);
        prg.recordPackHeader(&ca_record);
        sbt.addCallablesRecord(ca_record);
        return id;
    };

    // Callable programs for textures
    uint32_t constant_prg_id = setupCallable(tex_module, "__direct_callable__constant", "");
    uint32_t checker_prg_id = setupCallable(tex_module, "__direct_callable__checker", "");
    uint32_t bitmap_prg_id = setupCallable(tex_module, "__direct_callable__bitmap", "");

    // Callable programs for surfaces
    // Diffuse
    uint32_t diffuse_sample_bsdf_prg_id = setupCallable(surf_module, "__direct_callable__sample_diffuse", "__continuation_callable__bsdf_diffuse");
    uint32_t diffuse_pdf_prg_id = setupCallable(surf_module, "__direct_callable__pdf_diffuse", "");
    // Conductor
    uint32_t conductor_sample_bsdf_prg_id = setupCallable(surf_module, "__direct_callable__sample_conductor", "__continuation_callable__bsdf_conductor");
    uint32_t conductor_pdf_prg_id = setupCallable(surf_module, "__direct_callable__pdf_conductor", "");
    // Dielectric
    uint32_t dielectric_sample_bsdf_prg_id = setupCallable(surf_module, "__direct_callable__sample_dielectric", "__continuation_callable__bsdf_dielectric");
    uint32_t dielectric_pdf_prg_id = setupCallable(surf_module, "__direct_callable__pdf_dielectric", "");
    // Disney
    uint32_t disney_sample_bsdf_prg_id = setupCallable(surf_module, "__direct_callable__sample_disney", "__continuation_callable__bsdf_disney");
    uint32_t disney_pdf_prg_id = setupCallable(surf_module, "__direct_callable__pdf_disney", "");
    // Area emitter
    uint32_t area_emitter_prg_id = setupCallable(surf_module, "__direct_callable__area_emitter", "");

    // Callable program for direct sampling of area emitter
    uint32_t plane_sample_pdf_prg_id = setupCallable(light_sample_module, "__direct_callable__sample_plane", "__continuation_callable__pdf_plane");
    uint32_t sphere_sample_pdf_prg_id = setupCallable(light_sample_module, "__direct_callable__sample_sphere", "__continuation_callable__pdf_sphere");

    // Environment emitter
    textures.emplace("black", new ConstantTexture(Vec3f(0.0f), constant_prg_id));
    env = EnvironmentEmitter{ textures.at("black") };
    env.copyToDevice();

    // Miss program
    ProgramGroup ms_prg = pipeline.createMissProgram(context, ms_module, "__miss__envmap");
    pgMissRecord ms_record, ms_shadow_record;
    ms_prg.recordPackHeader(&ms_record);
    ms_record.data.env_data = env.devicePtr();

    ProgramGroup ms_shadow_prg = pipeline.createMissProgram(context, ms_module, "__miss__shadowmap");
    ms_shadow_prg.recordPackHeader(&ms_shadow_record);

    sbt.setMissRecord(ms_record, ms_shadow_record);

    // Hitgroup program
    // Plane
    auto plane_prg = pipeline.createHitgroupProgram(context, { ch_module, "__closesthit__custom" }, { is_module, "__intersection__plane" });
    auto plane_shadow_prg = pipeline.createHitgroupProgram(context, { ch_module, "__closesthit__shadow" }, { is_module, "__intersection__plane" });
    // Sphere
    auto sphere_prg = pipeline.createHitgroupProgram(context, { ch_module, "__closesthit__custom" }, { is_module, "__intersection__sphere" });
    auto sphere_shadow_prg = pipeline.createHitgroupProgram(context, { ch_module, "__closesthit__shadow" }, { is_module, "__intersection__sphere" });
    // Mesh
    auto mesh_prg = pipeline.createHitgroupProgram(context, { ch_module, "__closesthit__mesh" });
    auto mesh_shadow_prg = pipeline.createHitgroupProgram(context, { ch_module, "__closesthit__shadow" });

    struct Primitive {
        ShapePtr shape;
        MaterialPtr material;
        uint32_t sample_bsdf_id;
        uint32_t pdf_id;
    };

    uint32_t sbt_idx = 0, sbt_offset = 0, instance_id = 0;

    auto setupPrimitive = [&](ProgramGroup& prg, ProgramGroup& shadow_prg, const Primitive& p, const Matrix4f& m)
    {
        p.shape->copyToDevice();
        p.shape->setSbtIndex(sbt_idx);
        p.material->copyToDevice();

        pgHitgroupRecord record;
        prg.recordPackHeader(&record);
        pgHitgroupData record_data = {
            .shape_data = p.shape->devicePtr(), 
            .surface_info = 
            {
                .data = p.material->devicePtr(), 
                .sample_id = p.sample_bsdf_id, 
                .bsdf_id = p.sample_bsdf_id, 
                .pdf_id = p.pdf_id, 
                .type = p.material->surfaceType()
            }
        };

        record.data = record_data;

        pgHitgroupRecord shadow_record;
        shadow_prg.recordPackHeader(&shadow_record);
        shadow_record.data = record_data;

        sbt.addHitgroupRecord(record, shadow_record);
        sbt_idx += SBT::NRay;

        // Build GAS and add it to IAS
        ShapeInstance instance{p.shape->type(), p.shape, m};
        instance.allowCompaction();
        instance.buildAccel(context, stream);
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);

        ias.addInstance(instance);

        instance_id++;
        sbt_offset += SBT::NRay;
    };

    vector<AreaEmitterInfo> area_emitter_infos;
    auto setupAreaEmitter = [&](
        ProgramGroup& prg, ProgramGroup& shadow_prg,
        ShapePtr shape, AreaEmitterPtr area,
        const Matrix4f& m, uint32_t sample_pdf_id)
    {
        ASSERT(dynamic_pointer_cast<Plane>(shape) || dynamic_pointer_cast<Sphere>(shape),
            "The shape of area emitter must be a plane of sphere");

        shape->copyToDevice();
        shape->setSbtIndex(sbt_idx);
        area->copyToDevice();

        pgHitgroupRecord record;
        prg.recordPackHeader(&record);

        SurfaceInfo surface_info = {
            .data = area->devicePtr(),
            .sample_id = sample_pdf_id,
            .bsdf_id = area_emitter_prg_id,
            .pdf_id = sample_pdf_id,
            .type = SurfaceType::AreaEmitter
        };

        pgHitgroupData record_data = { shape->devicePtr(), surface_info };
        record.data = record_data;

        pgHitgroupRecord shadow_record;
        shadow_prg.recordPackHeader(&shadow_record);
        shadow_record.data = record_data;
        sbt_idx += SBT::NRay;

        sbt.addHitgroupRecord(record, shadow_record);

        // Build GAS and add it to IAS
        ShapeInstance instance{ shape->type(), shape, m };
        instance.allowCompaction();
        instance.buildAccel(context, stream);
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);
        
        ias.addInstance(instance);

        instance_id++;
        sbt_offset += SBT::NRay;

        AreaEmitterInfo area_emitter_info =
        {
            .shape_data = shape->devicePtr(), 
            .surface_info = surface_info, 
            .objToWorld = m, 
            .worldToObj = m.inverse(), 
            .sample_id = sample_pdf_id, 
            .pdf_id = sample_pdf_id
        };
        area_emitter_infos.emplace_back(area_emitter_info);
    };
    
    // Scene =====================================================
    textures.emplace("floor_checker", new CheckerTexture(Vec3f(0.3f), Vec3f(0.8f), 10, checker_prg_id));
    textures.emplace("green", new ConstantTexture(Vec3f(0.05f, 0.8f, 0.05f), constant_prg_id));
    textures.emplace("red", new ConstantTexture(Vec3f(0.8f, 0.05f, 0.05f), constant_prg_id));
    textures.emplace("gray", new ConstantTexture(Vec3f(0.75f), constant_prg_id));
    textures.emplace("white", new ConstantTexture(Vec3f(1.0f), constant_prg_id));

    materials.emplace("green_diffuse", new Diffuse(textures.at("green")));
    materials.emplace("red_diffuse", new Diffuse(textures.at("red")));
    materials.emplace("gray_diffuse", new Diffuse(textures.at("gray")));
    materials.emplace("checker_diffuse", new Diffuse(textures.at("floor_checker")));
    materials.emplace("metal", new Conductor(textures.at("white")));

    lights.emplace("ceiling", new AreaEmitter(textures.at("white"), 10.0f));

    shapes.emplace("wall_plane", new Plane(Vec2f(-25.0f), Vec2f(25.0f)));
    shapes.emplace("ceiling_plane", new Plane(Vec2f(-5.0f), Vec2f(5.0f)));
    auto bunny = make_shared<TriangleMesh>("resources/model/bunny.obj");
    bunny->smooth();
    shapes.emplace("bunny", bunny);

    // Floor 
    Primitive floor{ shapes.at("wall_plane"), materials.at("checker_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(plane_prg, plane_shadow_prg, floor, Matrix4f::translate(0, -25, 0));

    // Ceiling
    Primitive ceiling{ shapes.at("wall_plane"), materials.at("gray_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(plane_prg, plane_shadow_prg, ceiling, Matrix4f::translate(0, 25, 0));

    // Back wall 
    Primitive back{ shapes.at("wall_plane"), materials.at("gray_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(plane_prg, plane_shadow_prg, back, Matrix4f::translate(0, 0, 25) * Matrix4f::rotate(math::pi / 2.0f, { 1, 0, 0 }));

    // Right wall 
    Primitive right{ shapes.at("wall_plane"), materials.at("green_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(plane_prg, plane_shadow_prg, right, Matrix4f::translate(25, 0, 0)* Matrix4f::rotate(math::pi / 2.0f, { 0, 0, 1 }));

    // Left wall
    Primitive left{ shapes.at("wall_plane"), materials.at("red_diffuse"), diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };
    setupPrimitive(plane_prg, plane_shadow_prg, left, Matrix4f::translate(-25, 0, 0)* Matrix4f::rotate(math::pi / 2.0f, { 0, 0, 1 }));

    // Bunny 
    Primitive b{ shapes.at("bunny"), materials.at("metal"), conductor_sample_bsdf_prg_id, conductor_pdf_prg_id };
    setupPrimitive(mesh_prg, mesh_shadow_prg, b, Matrix4f::translate(0, -5, 0) * Matrix4f::scale(100));

    // Ceiling light
    setupAreaEmitter(plane_prg, plane_shadow_prg, shapes.at("ceiling_plane"), lights.at("ceiling"), Matrix4f::translate(0, 24.9f, 0), plane_sample_pdf_prg_id);

    // Copy light information to device
    CUDABuffer<AreaEmitterInfo> d_area_emitter_infos;
    d_area_emitter_infos.copyToDevice(area_emitter_infos);
    params.lights = d_area_emitter_infos.deviceData();
    params.num_lights = static_cast<uint32_t>(area_emitter_infos.size());

    CUDA_CHECK(cudaStreamCreate(&stream));
    ias.build(context, stream);
    sbt.createOnDevice();
    params.handle = ias.handle();
    pipeline.create(context);
    // Create pipeline for light vertices generation
    genlight_pipeline.createFromPrograms(context, pipeline.programs());
    d_params.allocate(sizeof(LaunchParams));

    // GUI setting
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    const char* glsl_version = "#version 150";
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(pgGetCurrentWindow()->windowPtr(), true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();

    params.frame++;
    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

    optixLaunch(
        (OptixPipeline)pipeline,
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

    result_bmp.copyFromDevice();
}

// ------------------------------------------------------------------
void App::draw()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Vertex connection and merging");

    ImGui::Text("Camera info:");
    ImGui::Text("Origin: %f %f %f", camera.origin().x(), camera.origin().y(), camera.origin().z());
    ImGui::Text("Lookat: %f %f %f", camera.lookat().x(), camera.lookat().y(), camera.lookat().z());

    ImGui::Text("Frame rate: %.3f ms/frame (%.2f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Frame index: %d", params.frame);

    ImGui::End();
    ImGui::Render();

    result_bmp.draw(0, 0);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// ------------------------------------------------------------------
void App::mousePressed(float x, float y, int button)
{
    camera_update = true;
}

// ------------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    camera_update = true;
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
    
}

// ------------------------------------------------------------------
void App::keyPressed(int key)
{
    if (key == Key::Q)
    {
        this->close();
        pgExit();
    }
}

// ------------------------------------------------------------------
void App::keyReleased(int key)
{

}



