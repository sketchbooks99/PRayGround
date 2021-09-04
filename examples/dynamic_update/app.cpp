#include "app.h"

// ----------------------------------------------------------------
void App::setup()
{
    // OptixDeviceContextの生成
    stream = 0;
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    context.setDeviceId(0);
    context.create();

    // パイプラインの設定
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(2);
    pipeline.setContinuationCallableDepth(2);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);

    // OptiXのModuleをCUDAファイルから生成
    Module raygen_module, miss_module, hitgroups_module, textures_module, surfaces_module;
    raygen_module = pipeline.createModuleFromCudaFile(context, "raygen.cu");
    miss_module = pipeline.createModuleFromCudaFile(context, "miss.cu");
    hitgroups_module = pipeline.createModuleFromCudaFile(context, "hitgroups.cu");
    textures_module = pipeline.createModuleFromCudaFile(context, "textures.cu");
    surfaces_module = pipeline.createModuleFromCudaFile(context, "surfaces.cu");

    // レンダリング結果を保存する用のBitmapを用意
    result_bitmap.allocate(Bitmap::Format::RGBA, pgGetWidth(), pgGetHeight());
    result_bitmap.allocateDevicePtr();
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;
    params.subframe_index = 0;
    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());

    // カメラの設定
    camera.setOrigin(make_float3(0.0f, 0.0f, 0.75f));
    camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFov(40.0f);
    camera.setFovAxis(Camera::FovAxis::Vertical);

    // Raygen プログラム用のデータ準備
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, raygen_module, "__raygen__pinhole");
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera =
    {
        .origin = camera.origin(), 
        .lookat = camera.lookat(), 
        .up = camera.up(), 
        .fov = camera.fov(), 
        .aspect = 1.0f
    };
    sbt.setRaygenRecord(raygen_record);

    // Callables プログラム用のデータ
    std::vector<EmptyRecord> callable_records(3, EmptyRecord{});

    // 単色テクスチャ用のプログラム
    auto [bitmap_prg, bitmap_prg_id] = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__bitmap", "");
    bitmap_prg.recordPackHeader(&callable_records[0]);
    sbt.addCallablesRecord(callable_records[0]);
    // チェッカーボード用のプログラム
    auto [checker_prg, checker_prg_id] = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__checker", "");
    checker_prg.recordPackHeader(&callable_records[1]);
    sbt.addCallablesRecord(callable_records[1]);

    // 環境マッピング用のテクスチャとデータを準備
    env_texture = make_shared<CheckerTexture>(make_float3(0.9f), make_float3(0.3f), 10.0f);
    env_texture->setProgramId(checker_prg_id);
    env_texture->copyToDevice();
    env = make_shared<EnvironmentEmitter>(env_texture);
    env->copyToDevice();

    // Miss プログラムの生成とSBTデータの用意
    ProgramGroup miss_prg = pipeline.createMissProgram(context, miss_module, "__miss__envmap");
    MissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env->devicePtr();
    sbt.setMissRecord(miss_record);

    // チェッカーボードテクスチャの準備
    checker_texture = make_shared<CheckerTexture>(make_float3(1.0f), make_float3(0.3f), 15);
    checker_texture->setProgramId(checker_prg_id);
    checker_texture->copyToDevice();

    // 面光源用のプログラムの生成
    auto [area_prg, area_prg_id] = pipeline.createCallablesProgram(context, surfaces_module, "__direct_callable__area_emitter", "");
    area_prg.recordPackHeader(&callable_records[2]);
    sbt.addCallablesRecord(callable_records[2]);

    // 面光源用のデータを準備
    area = make_shared<AreaEmitter>(checker_texture);
    area->copyToDevice();

    // Bunnyメッシュを.objから読み込む
    bunny = make_shared<TriangleMesh>("uv_bunny.obj");
    bunny->setSbtIndex(0);
    bunny->copyToDevice();

    // メッシュ用のプログラムの生成
    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__mesh");

    // Bunny用のHitgroupデータを用意し、SBTにデータをセットする
    HitgroupRecord bunny_record;
    mesh_prg.recordPackHeader(&bunny_record);
    bunny_record.data.shape_data = bunny->devicePtr();
    bunny_record.data.surface_data = area->devicePtr();
    bunny_record.data.surface_program_id = area_prg_id;
    bunny_record.data.surface_type = SurfaceType::AreaEmitter;
    sbt.addHitgroupRecord(bunny_record);
    
    // Bunny用のGeometry Acceleration Structure (GAS) を生成
    bunny_gas = GeometryAccel{ShapeType::Mesh};
    bunny_gas.addShape(bunny);
    bunny_gas.allowCompaction();
    bunny_gas.build(context, stream);
    
    // Shader Binding Table のデータをGPU上にコピーする
    sbt.createOnDevice();
    // レイトレーシング用のTraversableHandle を設定
    params.handle = bunny_gas.handle();
    // レイトレーシングパイプラインを生成
    pipeline.create(context);
    // cudaStreamの生成
    CUDA_CHECK(cudaStreamCreate(&stream));
    // GPU上にLaunchParamsデータ用の領域を確保
    d_params.allocate(sizeof(LaunchParams));
}

// ----------------------------------------------------------------
void App::update()
{
    float time = pgGetElapsedTime<float>();
    checker_texture->setColor1(make_float3(abs(sin(time))));
    checker_texture->copyToDevice();

    // 
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

// ----------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    /// @todo カメラの位置変更
}
