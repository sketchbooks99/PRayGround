#include "app.h"

// ----------------------------------------------------------------
void App::setup()
{
    // Initialization of device context.
    stream = 0;
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    context.setDeviceId(0);
    context.create();

    // Prepare pipeline
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(4);
    pipeline.setContinuationCallableDepth(4);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);
    pipeline.enableOpacityMap();

    // Create modules
    Module module = pipeline.createModuleFromCudaFile(context, "kernels.cu");

    // Prepare bitmap to store rendered results.
    result_bitmap.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());
    result_bitmap.allocateDevicePtr();
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;
    params.frame = 0;
    params.result_buffer = reinterpret_cast<Vec4u*>(result_bitmap.devicePtr());

    camera.setOrigin(0.0f, 0.0f, 0.75f);
    camera.setLookat(0.0f, 0.0f, 0.0f);
    camera.setUp(0.0f, 1.0f, 0.0f);
    camera.setFov(40.0f);
    camera.setFovAxis(Camera::FovAxis::Vertical);

    // Create raygen program and bind record;
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, module, "__raygen__pinhole");
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera = camera.getData();
    sbt.setRaygenRecord(raygen_record);

    // Callable関数とShader Binding TableにCallable関数用のデータを登録するLambda関数
    auto setupCallable = [&](const std::string& dc, const std::string& cc)
    {
        EmptyRecord callable_record = {};
        auto [prg, id] = pipeline.createCallablesProgram(context, module, dc, cc);
        prg.recordPackHeader(&callable_record);
        sbt.addCallablesRecord(callable_record);
        return id;
    };
    
    // Creating texture programs
    uint32_t checker_prg_id = setupCallable("__direct_callable__checker", "");

    // Prepare environment 
    env_texture = make_shared<CheckerTexture>(Vec3f(0.9f), Vec3f(0.3f), 10.0f, checker_prg_id);
    env_texture->copyToDevice();
    env = make_shared<EnvironmentEmitter>(env_texture);
    env->copyToDevice();

    ProgramGroup miss_prg = pipeline.createMissProgram(context, module, "__miss__envmap");
    MissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env->devicePtr();
    sbt.setMissRecord({ miss_record });

    // Preparing textures
    checker_texture = make_shared<CheckerTexture>(Vec3f(1.0f), Vec3f(0.3f), 15, checker_prg_id);
    checker_texture->copyToDevice();

    // Preparing materials and program id
    area = make_shared<AreaEmitter>(SurfaceCallableID{ 0, 0, 0 }, checker_texture);
    area->copyToDevice();

    // Load mesh from .obj file
    bunny = make_shared<TriangleMesh>("resources/model/uv_bunny.obj");
    bunny->setSbtIndex(0);
    bunny->copyToDevice();

    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh");

    HitgroupRecord bunny_record;
    mesh_prg.recordPackHeader(&bunny_record);
    bunny_record.data = 
    {
        .shape_data = bunny->devicePtr(),
        .texture = checker_texture->getData()
    };
    sbt.addHitgroupRecord({ bunny_record });
    
    // Build GAS
    gas = GeometryAccel{ShapeType::Mesh};
    gas.addShape(bunny);
    gas.allowCompaction();
    gas.build(context, stream);
    
    // Create sbt and pipeline to launch ray
    sbt.createOnDevice();
    params.handle = gas.handle();
    pipeline.create(context);
    CUDA_CHECK(cudaStreamCreate(&stream));
    d_params.allocate(sizeof(LaunchParams));
}

// ----------------------------------------------------------------
void App::update()
{
    float time = pgGetElapsedTimef();
    checker_texture->setColor1(make_float3(abs(sin(time))));
    checker_texture->copyToDevice();

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

    CUDA_CHECK(cudaSetDevice(context.deviceId()));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    result_bitmap.copyFromDevice();
}

// ----------------------------------------------------------------
void App::draw()
{
    result_bitmap.draw(0, 0);
}
