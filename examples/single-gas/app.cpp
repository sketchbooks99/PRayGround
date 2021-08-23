#include "app.h"
#include <oprt/core/interaction.h>

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
    film.bitmapAt("result")->allocateDeviceData();
    film.floatBitmapAt("accum")->allocateDeviceData();
    params.width = film.width();
    params.height = film.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;
    params.subframe_index = 0;
    params.accum_buffer = reinterpret_cast<float4*>(film.floatBitmapAt("accum")->devicePtr());
    params.result_buffer = reinterpret_cast<uchar4*>(film.bitmapAt("result")->devicePtr());

    camera.setOrigin(make_float3(0.0f, 0.0f, 50.0f));
    camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFov(40.0f);
    camera.setFovAxis(Camera::FovAxis::Vertical);

    // Create raygen program and bind record;
    pipeline.createRaygenProgram(context, raygen_module, "__raygen__pinhole");
    RaygenRecord raygen_record;
    raygen_record.data.camera.origin = camera.origin();
    raygen_record.data.camera.lookat = camera.lookat();
    raygen_record.data.camera.up = camera.up();
    raygen_record.data.camera.fov = camera.fov();
    raygen_record.data.camera.aspect = 1.0f;
    //sbt.setRaygenRecord(raygen_record);
    CUDABuffer<RaygenRecord> d_raygen_record;
    d_raygen_record.copyToDevice(&raygen_record, sizeof(RaygenRecord));
    sbt.raygenRecord = d_raygen_record.devicePtr();
    pipeline.bindRaygenRecord(&raygen_record);

    // SBT record for callable programs
    std::vector<EmptyRecord> callable_records(3, EmptyRecord{});
    //sbt.addCallablesRecord(callable_record);

    // Creating texture programs
    uint32_t checker_prg_id = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__checker", "");
    uint32_t bitmap_prg_id = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__bitmap", "");
    pipeline.bindCallablesRecord(&callable_records[0], 0);
    pipeline.bindCallablesRecord(&callable_records[1], 1);

    // Prepare environment 
    env = make_shared<EnvironmentEmitter>("016_hdrmaps_com_free.exr");
    env->texture()->setProgramId(bitmap_prg_id);
    env->copyToDevice();

    pipeline.createMissProgram(context, miss_module, "__miss__envmap");
    MissRecord miss_record;
    miss_record.data.env_data = env->devicePtr();
    //sbt.setMissRecord(miss_record);
    pipeline.bindMissRecord(&miss_record, 0);

    CUDABuffer<MissRecord> d_miss_record;
    d_miss_record.copyToDevice(&miss_record, sizeof(MissRecord));
    sbt.missRecordBase = d_miss_record.devicePtr();
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof(MissRecord));

    // Preparing textures
    texture = make_shared<CheckerTexture>(make_float3(1.0f), make_float3(0.3f), 5);

    // Creating material and emitter programs
    uint32_t area_emitter_prg_id = pipeline.createCallablesProgram(context, surfaces_module, "__direct_callable__area_emitter", "");
    pipeline.bindCallablesRecord(&callable_records[2], 2);

    CUDABuffer<EmptyRecord> d_callable_records;
    d_callable_records.copyToDevice(callable_records);
    sbt.callablesRecordBase = d_callable_records.devicePtr();
    sbt.callablesRecordCount = static_cast<uint32_t>(callable_records.size());
    sbt.callablesRecordStrideInBytes = static_cast<uint32_t>(sizeof(EmptyRecord));

    // Preparing materials and program id
    area = make_shared<AreaEmitter>(make_float3(1.0f), 15.0f);
    area->setProgramId(area_emitter_prg_id);
    area->copyToDevice();

    // Prepare cornel box and construct its gas and instance
    bunny = make_shared<TriangleMesh>("bunny.obj");
    bunny->setSbtIndex(0);
    bunny->attachSurface(area);
    bunny->copyToDevice();

    pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__mesh");

    HitgroupRecord bunny_record;
    bunny_record.data.shape_data = bunny->devicePtr();
    bunny_record.data.surface_data = bunny->surfaceDevicePtr();
    bunny_record.data.surface_program_id = area->programId();
    pipeline.bindHitgroupRecord(&bunny_record, 0);
    //sbt.addHitgroupRecord(bunny_record);

    CUDABuffer<HitgroupRecord> d_hitgroup_records;
    d_hitgroup_records.copyToDevice(&bunny_record, sizeof(HitgroupRecord));
    sbt.hitgroupRecordBase = d_hitgroup_records.devicePtr();
    sbt.hitgroupRecordCount = 1;
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitgroupRecord));

    gas = GeometryAccel{GeometryAccel::Type::Mesh};
    gas.addShape(bunny);
    gas.allowCompaction();
    gas.build(context);

    // Build IAS
    //sbt.createOnDevice();
    params.handle = gas.handle();
    pipeline.create(context);
    CUDA_CHECK(cudaStreamCreate(&stream));
    d_params.allocate(sizeof(LaunchParams));
}

// ----------------------------------------------------------------
void App::update()
{
    film.bitmapAt("result")->allocateDeviceData();
    film.floatBitmapAt("accum")->allocateDeviceData();
    params.result_buffer = reinterpret_cast<uchar4*>(film.bitmapAt("result")->devicePtr());
    params.accum_buffer = reinterpret_cast<float4*>(film.floatBitmapAt("accum")->devicePtr());

    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

    OPTIX_CHECK(optixLaunch(
        static_cast<OptixPipeline>(pipeline),
        stream,
        d_params.devicePtr(),
        sizeof(LaunchParams),
        &sbt,
        params.width,
        params.height,
        1
    ));

    CUDA_CHECK(cudaSetDevice(context.deviceId()));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    //film.bitmapAt("result")->copyFromDevice();
}

// ----------------------------------------------------------------
void App::draw()
{
    Message(MSG_WARNING, "Draw called");
    //film.bitmapAt("result")->draw(0, 0, film.width(), film.height());
}

// ----------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    
}
