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
    raygen_module = pipeline.createModuleFromCudaFile(context, "raygen.cu");
    miss_module = pipeline.createModuleFromCudaFile(context, "miss.cu");
    hitgroups_module = pipeline.createModuleFromCudaFile(context, "hitgroups.cu");
    textures_module = pipeline.createModuleFromCudaFile(context, "textures.cu");
    surfaces_module = pipeline.createModuleFromCudaFile(context, "surfaces.cu");

    // Prepare film to store rendered results.
    film = Film(1024, 1024);
    film.addBitmap("result", Bitmap::Format::RGBA);
    film.addFloatBitmap("accum", FloatBitmap::Format::RGBA);
    film.bitmapAt("result")->allocateDevicePtr();
    film.floatBitmapAt("accum")->allocateDevicePtr();
    params.width = film.width();
    params.height = film.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;
    params.subframe_index = 0;
    params.accum_buffer = reinterpret_cast<float4*>(film.floatBitmapAt("accum")->devicePtr());
    params.result_buffer = reinterpret_cast<uchar4*>(film.bitmapAt("result")->devicePtr());

    camera.setOrigin(make_float3(0.0f, 0.0f, 0.75f));
    camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFov(40.0f);
    camera.setFovAxis(Camera::FovAxis::Vertical);

    // Create raygen program and bind record;
    pipeline.createRaygenProgram(context, raygen_module, "__raygen__pinhole");
    RaygenRecord raygen_record;
    pipeline.bindRaygenRecord(&raygen_record);
    raygen_record.data.camera.origin = camera.origin();
    raygen_record.data.camera.lookat = camera.lookat();
    raygen_record.data.camera.up = camera.up();
    raygen_record.data.camera.fov = camera.fov();
    raygen_record.data.camera.aspect = 1.0f;
    sbt.setRaygenRecord(raygen_record);

    // SBT record for callable programs
    std::vector<EmptyRecord> callable_records(3, EmptyRecord{});

    // Creating texture programs
    uint32_t bitmap_prg_id = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__bitmap", "");
    uint32_t checker_prg_id = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__checker", "");
    pipeline.bindCallablesRecord(&callable_records[0], 0);
    pipeline.bindCallablesRecord(&callable_records[1], 1);
    sbt.addCallablesRecord(callable_records[0]);
    sbt.addCallablesRecord(callable_records[1]);

    // Prepare environment 
    env_texture = make_shared<FloatBitmapTexture>("env2.jpg");
    env_texture->setProgramId(bitmap_prg_id);
    env_texture->copyToDevice();
    env = make_shared<EnvironmentEmitter>(env_texture);
    env->copyToDevice();

    pipeline.createMissProgram(context, miss_module, "__miss__envmap");
    MissRecord miss_record;
    pipeline.bindMissRecord(&miss_record, 0);
    miss_record.data.env_data = env->devicePtr();
    sbt.setMissRecord(miss_record);

    // Preparing textures
    checker_texture = make_shared<CheckerTexture>(make_float3(1.0f), make_float3(0.3f), 15);
    checker_texture->setProgramId(checker_prg_id);
    checker_texture->copyToDevice();

    // Creating material and emitter programs
    uint32_t area_prg_id = pipeline.createCallablesProgram(context, surfaces_module, "__direct_callable__area_emitter", "");
    pipeline.bindCallablesRecord(&callable_records[2], 2);
    sbt.addCallablesRecord(callable_records[2]);

    // Preparing materials and program id
    area = make_shared<AreaEmitter>(checker_texture);
    area->copyToDevice();

    // Load mesh from .obj file
    bunny = make_shared<TriangleMesh>("uv_bunny.obj");
    bunny->setSbtIndex(0);
    bunny->copyToDevice();

    pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__mesh");

    HitgroupRecord bunny_record;
    pipeline.bindHitgroupRecord(&bunny_record, 0);
    bunny_record.data.shape_data = bunny->devicePtr();
    bunny_record.data.surface_data = area->devicePtr();
    bunny_record.data.surface_program_id = area_prg_id;
    bunny_record.data.surface_type = SurfaceType::AreaEmitter;
    sbt.addHitgroupRecord(bunny_record);
    
    // Build GAS
    gas = GeometryAccel{GeometryAccel::Type::Mesh};
    gas.addShape(bunny);
    gas.allowCompaction();
    gas.build(context);
    
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
    float time = pgGetElapsedTime<float>();
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

    film.bitmapAt("result")->copyFromDevice();
}

// ----------------------------------------------------------------
void App::draw()
{
    film.bitmapAt("result")->draw(0, 0);
}

// ----------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    /// @todo カメラの位置変更
}
