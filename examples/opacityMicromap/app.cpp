#include "app.h"
void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.deviceData());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.deviceData());

    CUDA_SYNC_CHECK();
}

void App::handleCameraUpdate()
{
    if (!is_camera_updated)
        return;
    is_camera_updated = false;

    RaygenRecord* rg_record = reinterpret_cast<RaygenRecord*>(sbt.deviceRaygenRecordPtr());
    RaygenData rg_data;
    rg_data.camera = camera.getData();

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(&rg_record->data),
        &rg_data, sizeof(pgRaygenData<Camera>),
        cudaMemcpyHostToDevice
    ));

    initResultBufferOnDevice();
}

// ------------------------------------------------------------------
void App::setup()
{
    // Initialize CUDA
    stream = 0;
    CUDA_CHECK(cudaFree(0));

    // Initialize device context
    OPTIX_CHECK(optixInit());
    context.disableValidation();
    context.create();

    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(0);
    pipeline.setContinuationCallableDepth(0);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);
    // Must be called
    pipeline.enableOpacityMap();
    pipeline.setTraversableGraphFlags(OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS);

    // Create module
    Module module;
    module = pipeline.createModuleFromCudaFile(context, "kernels.cu");

    const int32_t width = pgGetWidth();
    const int32_t height = pgGetHeight();
    result_bmp.allocate(PixelFormat::RGBA, width, height);
    result_bmp.allocateDevicePtr();
    accum_bmp.allocate(PixelFormat::RGBA, width, height);
    accum_bmp.allocateDevicePtr();

    // Configuration of launch parameters
    params.width = width;
    params.height = height;
    params.samples_per_launch = 1;
    params.frame = 0;
    params.max_depth = 5;
    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.deviceData());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.deviceData());

    initResultBufferOnDevice();

    // Camera settings
    camera.setOrigin(0, 30, 0);
    camera.setLookat(0, 0, 0);
    camera.setUp(0, 1, 0);
    camera.setFov(40);
    camera.setAspect((float)width / height);
    camera.enableTracking(pgGetCurrentWindow());

    // Raygen program
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, module, "__raygen__pinhole");
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera = camera.getData();
    sbt.setRaygenRecord(raygen_record);

    // Callable programs
    auto setupCallable = [&](const Module& module, const std::string& dc, const std::string& cc) -> uint32_t
    {
        EmptyRecord callable_record = {};
        auto [prg, id] = pipeline.createCallablesProgram(context, module, dc, cc);
        prg.recordPackHeader(&callable_record);
        sbt.addCallablesRecord(callable_record);
        return id;
    };

    // Textures
    uint32_t bitmap_prg_id = setupCallable(module, "__direct_callable__bitmap", "");
    uint32_t constant_prg_id = setupCallable(module, "__direct_callable__constant", "");
    uint32_t checker_prg_id = setupCallable(module, "__direct_callable__checker", "");

    // Surfaces
    // Diffuse
    SurfaceCallableID diffuse_id = {};
    diffuse_id.sample = setupCallable(module, "__direct_callable__sample_diffuse", "");
    diffuse_id.bsdf = setupCallable(module, "__direct_callable__bsdf_diffuse", "");
    diffuse_id.pdf = setupCallable(module, "__direct_callable__pdf_diffuse", "");

    // Area emitter
    uint32_t area_emitter_prg_id = setupCallable(module, "__direct_callable__area_emitter", "");

    // Setup environment emitter
    auto env_texture = make_shared<ConstantTexture>(Vec3f(0.5f), constant_prg_id);
    env_texture->copyToDevice();
    env = EnvironmentEmitter{ env_texture };
    env.copyToDevice();

    // Miss program
    ProgramGroup miss_prg = pipeline.createMissProgram(context, module, "__miss__envmap");
    MissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env.devicePtr();
    sbt.setMissRecord({ miss_record });

    // Hitgroup program
    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh", "", "__anyhit__opacity");
    ProgramGroup mesh_opaque_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh");

    uint32_t sbt_idx = 0;
    uint32_t sbt_offset = 0;
    uint32_t instance_id = 0;

    using SurfaceP = variant<shared_ptr<Material>, shared_ptr<AreaEmitter>>;
    auto addHitgroupRecord = [&](ProgramGroup& prg, shared_ptr<Shape> shape, SurfaceP surface, shared_ptr<Texture> opacity_texture = nullptr) -> void
    {
        const bool is_mat = holds_alternative<shared_ptr<Material>>(surface);
        if (opacity_texture) opacity_texture->copyToDevice();

        // Copy data to GPU
        shape->copyToDevice();
        shape->setSbtIndex(sbt_idx);
        if (is_mat)
            get<shared_ptr<Material>>(surface)->copyToDevice();
        else
            get<shared_ptr<AreaEmitter>>(surface)->copyToDevice();

        // Register data to shader binding table
        HitgroupRecord record;
        prg.recordPackHeader(&record);
        record.data = {
            .shape_data = shape->devicePtr(),
            .surface_info = {
                .data = is_mat ? get<shared_ptr<Material>>(surface)->devicePtr() : get<shared_ptr<AreaEmitter>>(surface)->devicePtr(),
                .callable_id = is_mat ? get<shared_ptr<Material>>(surface)->surfaceCallableID() : get<shared_ptr<AreaEmitter>>(surface)->surfaceCallableID(),
                .type = is_mat ? get<shared_ptr<Material>>(surface)->surfaceType() : SurfaceType::AreaEmitter
            },
            .opacity_texture = opacity_texture ? opacity_texture->getData() : Texture::Data{nullptr, bitmap_prg_id}
        };

        sbt.addHitgroupRecord({ record });
        sbt_idx++;
    };

    // function to determine opacity states according to texture coordinates on micro triangles
    auto omm_function = [](const OpacityMicromap::MicroBarycentrics& bc, const Vec2f& uv0, const Vec2f& uv1, const Vec2f& uv2) -> int
    {
        // Calculate texcoords for each micro triangles in opacity micromap
        const Vec2f micro_uv0 = barycentricInterop(uv0, uv1, uv2, bc.uv0);
        const Vec2f micro_uv1 = barycentricInterop(uv0, uv1, uv2, bc.uv1);
        const Vec2f micro_uv2 = barycentricInterop(uv0, uv1, uv2, bc.uv1);

        const bool in_circle0 = (length(micro_uv0 - 0.5f)) < 0.25f;
        const bool in_circle1 = (length(micro_uv1 - 0.5f)) < 0.25f;
        const bool in_circle2 = (length(micro_uv2 - 0.5f)) < 0.25f;

        if (in_circle0 && in_circle1 && in_circle2)
            return OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
        else
            return OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
    };

    // Load bitmap texture to determine opacity micro map
    auto opacity_bmp = make_shared<BitmapTexture>("resources/image/PRayGround_black.png", bitmap_prg_id);
    opacity_bmp->copyToDevice();

    // Create mesh with the size of opacity bitmap
    Vec2f mesh_size((float)opacity_bmp->width() / opacity_bmp->height(), 1.0f);
    auto mesh = make_shared<PlaneMesh>(mesh_size, Vec2ui(1,1), Axis::Z);
    mesh->setSbtIndex(0);
    // Set up opacity bitmap 
    mesh->setupOpacitymap(context, stream, 8, OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE, opacity_bmp, OPTIX_OPACITY_MICROMAP_FLAG_NONE);
    mesh->copyToDevice();

    auto diffuse = make_shared<Diffuse>(diffuse_id, opacity_bmp);
    diffuse->copyToDevice();

    HitgroupRecord hitgroup_record;
    mesh_prg.recordPackHeader(&hitgroup_record);
    hitgroup_record.data = {
        .shape_data = mesh->devicePtr(),
        .surface_info = {
            .data = diffuse->devicePtr(),
            .callable_id = diffuse->surfaceCallableID(),
            .type = diffuse->surfaceType()
        },
        .opacity_texture = { .data = opacity_bmp->devicePtr(), .prg_id = bitmap_prg_id }
    };

    sbt.addHitgroupRecord({ hitgroup_record });

    GeometryAccel gas{ ShapeType::Mesh };
    gas.addShape(mesh);
    gas.allowCompaction();
    gas.build(context, stream);

    CUDA_CHECK(cudaStreamCreate(&stream));
    sbt.createOnDevice();
    params.handle = gas.handle();
    pipeline.create(context);
    d_params.allocate(sizeof(LaunchParams));
}

// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();

    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);
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

    params.frame++;

    result_bmp.copyFromDevice();
}

// ------------------------------------------------------------------
void App::draw()
{
    result_bmp.draw();
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



