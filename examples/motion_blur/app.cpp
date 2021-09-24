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

    instance_accel = InstanceAccel{ InstanceAccel::Type::Instances };
    instance_accel.allowUpdate();
    OptixMotionOptions motion_options;
    motion_options.numKeys = 2;
    motion_options.timeBegin = 0.0f;
    motion_options.timeEnd = 1.0f;
    motion_options.flags = OPTIX_MOTION_FLAG_NONE;
    instance_accel.setMotionOptions(motion_options);

    // パイプラインの設定
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(2);
    pipeline.setContinuationCallableDepth(2);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);
    // Motion blurを有効にする
    pipeline.enableMotionBlur();

    // OptiXのModuleをCUDAファイルから生成
    Module raygen_module, miss_module, hitgroups_module, textures_module;
    raygen_module = pipeline.createModuleFromCudaFile(context, "cuda/raygen.cu");
    miss_module = pipeline.createModuleFromCudaFile(context, "cuda/miss.cu");
    hitgroups_module = pipeline.createModuleFromCudaFile(context, "cuda/hitgroups.cu");
    textures_module = pipeline.createModuleFromCudaFile(context, "cuda/textures.cu");

    // レンダリング結果を保存する用のBitmapを用意
    result_bitmap.allocate(Bitmap::Format::RGBA, pgGetWidth(), pgGetHeight());
    result_bitmap.allocateDevicePtr();
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.subframe_index = 0;
    params.light.pos = make_float3(0.0f, 9.9f, 0.0f);
    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());

    // カメラの設定
    camera.setOrigin(make_float3(0.0f, 0.0f, 40.0f));
    camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFov(40.0f);
    camera.setAspect(params.width / params.height);
    float3 U, V, W;
    camera.UVWFrame(U, V, W);

    // Raygen プログラム用のデータ準備
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

    // テクスチャ用のCallableプログラム
    uint32_t constant_prg_id = setupCallable(textures_module, "__direct_callable__constant", "");
    uint32_t checker_prg_id = setupCallable(textures_module, "__direct_callable__checker", "");

    // 環境マッピング用のテクスチャとデータを準備
    auto env_color = make_shared<ConstantTexture>(make_float3(0.5f), constant_prg_id);
    env_color->copyToDevice();
    auto env = EnvironmentEmitter{env_color};
    env.copyToDevice();

    // Miss プログラムの生成とSBTデータの用意
    ProgramGroup miss_prg = pipeline.createMissProgram(context, miss_module, "__miss__envmap");
    MissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env.devicePtr();
    sbt.setMissRecord(miss_record);

    uint32_t sbt_offset = 0;
    uint32_t sbt_idx = 0;
    uint32_t instance_id = 0;

    // ShapeInstanceは1つのShapeのみ保持する前提でShapeInstanceをセットアップするLambda関数
    auto setupShapeInstance = [&](ProgramGroup hitgroup_prg, ShapeInstance& instance, shared_ptr<Texture> texture)
    {
        auto shape = instance.shapes()[0];

        shape->setSbtIndex(sbt_idx);
        // データをGPU上に用意
        shape->copyToDevice();
        texture->copyToDevice();

        HitgroupRecord record;
        hitgroup_prg.recordPackHeader(&record);
        record.data = 
        {
            .shape_data = shape->devicePtr(),
            .tex_data = 
            {
                .data = texture->devicePtr(), 
                .prg_id = texture->programId()
            }
        };

        sbt.addHitgroupRecord(record);

        instance.allowCompaction();
        instance.allowUpdate();
        instance.allowRandomVertexAccess();
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);
        instance.buildAccel(context, stream);

        instance_accel.addInstance(instance);

        instance_id++;
        sbt_offset += MotionBlurSBT::NRay;
        sbt_idx++;
    };

    auto plane_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__plane", "__intersection__plane");
    auto sphere_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__sphere", "__intersection__sphere");
    auto mesh_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__mesh");

    // Cornel box
    {
        // Cornel box用のテクスチャを用意
        auto floor_checker = make_shared<CheckerTexture>(make_float3(0.9f), make_float3(0.3f), 10, checker_prg_id);
        auto red = make_shared<ConstantTexture>(make_float3(0.8f, 0.05f, 0.05f), constant_prg_id);
        auto green = make_shared<ConstantTexture>(make_float3(0.05f, 0.8f, 0.05f), constant_prg_id);
        auto white = make_shared<ConstantTexture>(make_float3(0.8f), constant_prg_id);

        ShapeInstance floor {
            ShapeType::Custom, 
            make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f)),
            Matrix4f::translate(make_float3(0.0f, -10.0f, 0.0f))
        };

        ShapeInstance ceiling {
            ShapeType::Custom, 
            make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f)),
            Matrix4f::translate(make_float3(0.0f, 10.0f, 0.0f))
        };

        ShapeInstance back {
            ShapeType::Custom, 
            make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f)),
            // x軸中心に90度回転
            Matrix4f::translate(make_float3(0.0f, 0.0f, -10.0f)) * Matrix4f::rotate(math::pi / 2, {1.0f, 0.0f, 0.0f}) 
        };

        ShapeInstance right_wall {
            ShapeType::Custom, 
            make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f)),
            // z軸中心に90度回転
            Matrix4f::translate(make_float3(10.0f, 0.0f, 0.0f)) * Matrix4f::rotate(math::pi / 2, {0.0f, 0.0f, 1.0f})
        };

        ShapeInstance left_wall {
            ShapeType::Custom, 
            make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f)),
            // z軸中心に90度回転
            Matrix4f::translate(make_float3(-10.0f, 0.0f, 0.0f)) * Matrix4f::rotate(math::pi / 2, {0.0f, 0.0f, 1.0f})
        };

        setupShapeInstance(plane_prg, floor, floor_checker);
        setupShapeInstance(plane_prg, ceiling, white);
        setupShapeInstance(plane_prg, right_wall, green);
        setupShapeInstance(plane_prg, left_wall, red);
        setupShapeInstance(plane_prg, back, white);
    }

    // Sphere
    {
        auto sphere_texture = make_shared<ConstantTexture>(make_float3(0.8f), constant_prg_id);
        sphere_texture->copyToDevice();

        auto sphere = make_shared<Sphere>();
        sphere->setSbtIndex(sbt_idx);
        sphere->copyToDevice();

        HitgroupRecord record;
        sphere_prg.recordPackHeader(&record);
        record.data = 
        {
            .shape_data = sphere->devicePtr(),
            .tex_data = 
            {
                .data = sphere_texture->devicePtr(), 
                .prg_id = sphere_texture->programId()
            }
        };

        sbt.addHitgroupRecord(record);

        // 球体用のGASを用意
        GeometryAccel sphere_gas{ShapeType::Custom};
        sphere_gas.addShape(sphere);
        sphere_gas.allowCompaction();
        sphere_gas.allowUpdate();
        sphere_gas.build(context, stream);

        // Motion blur用の開始と終了点における変換行列を用意
        Matrix4f begin_matrix = Matrix4f::translate({5.0f, 0.0f, 0.0f}) * Matrix4f::scale(3.0f);
        Matrix4f end_matrix = Matrix4f::translate({-5.0f, 0.0f, 0.0f}) * Matrix4f::scale(3.0f);

        // Matrix motion 用のTransformを用意
        Transform matrix_transform{TransformType::MatrixMotion};
        matrix_transform.setChildHandle(sphere_gas.handle());
        matrix_transform.setMotionOptions(instance_accel.motionOptions());
        matrix_transform.setMatrixMotionTransform(begin_matrix, end_matrix);
        // childHandleからTransformのTraversableHandleを生成
        matrix_transform.buildHandle(context);

        // Instanceの生成
        Instance instance;
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);
        instance.setTraversableHandle(matrix_transform.handle());

        instance_accel.addInstance(instance);
    }
    
    // Shader Binding Table のデータをGPU上に生成
    sbt.createOnDevice();
    // レイトレーシング用のTraversableHandle を設定
    instance_accel.build(context, stream);
    params.handle = instance_accel.handle();
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
    params.subframe_index++;
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

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    result_bitmap.copyFromDevice();
}

// ----------------------------------------------------------------
void App::draw()
{
    result_bitmap.draw(0, 0);
}