#include "app.h"

void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.devicePtr());

    CUDA_SYNC_CHECK();
}

void App::handleCameraUpdate()
{
    if (!is_camera_updated)
        return;
    is_camera_updated = false;

    pgRaygenRecord<Camera>* rg_record = reinterpret_cast<pgRaygenRecord<Camera>*>(sbt.deviceRaygenRecordPtr());
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
    params.samples_per_launch = 4;
    params.frame = 0;
    params.max_depth = 5;
    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.devicePtr());

    initResultBufferOnDevice();

    // Camera settings
    camera.setOrigin(0, 0, -5);
    camera.setLookat(0, 0, 0);
    camera.setUp(0, 1, 0);
    camera.setFov(40);
    camera.setAspect((float)width / height);
    camera.enableTracking(pgGetCurrentWindow());

    // Raygen program
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, module, "__raygen__pinhole");
    pgRaygenRecord<Camera> raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera = camera.getData();
    sbt.setRaygenRecord(raygen_record);

    // Setup environment emitter
    auto env_texture = make_shared<FloatBitmapTexture>("resources/image/drackenstein_quarry_4k.exr", 2);
    env_texture->copyToDevice();
    EnvironmentEmitter env{ env_texture };
    env.copyToDevice();

    // Miss program
    ProgramGroup miss_prg = pipeline.createMissProgram(context, module, "__miss__envmap");
    pgMissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env.devicePtr();
    sbt.setMissRecord({ miss_record });

    // Hitgroup program
    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh", "", "__anyhit__opacity");

    // Load opacity texture
    std::vector<Vec3f> vertices = { Vec3f(-1, 1, 0), Vec3f(-1, -1, 0), Vec3f(1, -1, 0), Vec3f(1, 1, 0) };
    std::vector<Vec3f> normals = { Vec3f(0, 0, -1), Vec3f(0, 0, -1), Vec3f(0, 0, -1), Vec3f(0, 0, -1) };
    std::vector<Vec2f> texcoords = { Vec2f(0, 0), Vec2f(0, 1), Vec2f(1, 1), Vec2f(1, 0) };
    std::vector<Face> faces = {
        {Vec3i(0, 1, 2), Vec3i(0, 1, 2), Vec3i(0, 1, 2)},
        {Vec3i(0, 2, 3), Vec3i(0, 2, 3), Vec3i(0, 2, 3)}
    };

    auto mesh = make_shared<TriangleMesh>(vertices, faces, normals, texcoords);
    mesh->copyToDevice();

    auto diffuse = make_shared<Diffuse>(SurfaceCallableID{}, make_shared<ConstantTexture>(Vec3f(1.0f), 0));
    diffuse->copyToDevice();

    pgHitgroupRecord hitgroup_record;
    mesh_prg.recordPackHeader(&hitgroup_record);
    hitgroup_record.data = {
        .shape_data = mesh->devicePtr(),
        .surface_info = {
            .data = diffuse->devicePtr(),
            .callable_id = diffuse->surfaceCallableID(),
            .type = diffuse->surfaceType()
        }
    };

    sbt.addHitgroupRecord({ hitgroup_record });

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

    mesh->setupOpacitymap(context, 2, OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE, omm_function, OPTIX_OPACITY_MICROMAP_FLAG_NONE);

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



