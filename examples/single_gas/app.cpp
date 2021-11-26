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

    // Create modules
    Module raygen_module, miss_module, hitgroups_module, textures_module, surfaces_module;
    raygen_module = pipeline.createModuleFromCudaFile(context, "cuda/raygen.cu");
    miss_module = pipeline.createModuleFromCudaFile(context, "cuda/miss.cu");
    hitgroups_module = pipeline.createModuleFromCudaFile(context, "cuda/hitgroups.cu");
    textures_module = pipeline.createModuleFromCudaFile(context, "cuda/textures.cu");

    // Prepare bitmap to store rendered results.
    result_bitmap.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());
    result_bitmap.allocateDevicePtr();
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;
    params.subframe_index = 0;
    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());

    camera.setOrigin(make_float3(0.0f, 0.0f, 0.75f));
    camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFov(40.0f);
    camera.setFovAxis(Camera::FovAxis::Vertical);

    float3 U, V, W;
    camera.UVWFrame(U, V, W);

    // Create raygen program and bind record;
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, raygen_module, "__raygen__pinhole");
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

    // Callable関数とShader Binding TableにCallable関数用のデータを登録するLambda関数
    auto setupCallable = [&](const Module& module, const std::string& dc, const std::string& cc)
    {
        EmptyRecord callable_record = {};
        auto [prg, id] = pipeline.createCallablesProgram(context, module, dc, cc);
        prg.recordPackHeader(&callable_record);
        sbt.addCallablesRecord(callable_record);
        return id;
    };
    
    // Creating texture programs
    uint32_t checker_prg_id = setupCallable(textures_module, "__direct_callable__checker", "");

    // Prepare environment 
    env_texture = make_shared<CheckerTexture>(make_float3(0.9f), make_float3(0.3f), 10.0f, checker_prg_id);
    env_texture->copyToDevice();
    env = make_shared<EnvironmentEmitter>(env_texture);
    env->copyToDevice();

    ProgramGroup miss_prg = pipeline.createMissProgram(context, miss_module, "__miss__envmap");
    MissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env->devicePtr();
    sbt.setMissRecord(miss_record);

    // Preparing textures
    checker_texture = make_shared<CheckerTexture>(make_float3(1.0f), make_float3(0.3f), 15, checker_prg_id);
    checker_texture->copyToDevice();

    // Preparing materials and program id
    area = make_shared<AreaEmitter>(checker_texture);
    area->copyToDevice();

    // Load mesh from .obj file
    bunny = make_shared<TriangleMesh>("resources/model/uv_bunny.obj");
    bunny->setSbtIndex(0);
    bunny->copyToDevice();

    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__mesh");

    HitgroupRecord bunny_record;
    mesh_prg.recordPackHeader(&bunny_record);
    bunny_record.data = 
    {
        .shape_data = bunny->devicePtr(),
        .tex_data = 
        {
            .data = checker_texture->devicePtr(), 
            .prg_id = checker_prg_id
        }
    };
    sbt.addHitgroupRecord(bunny_record);
    
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
