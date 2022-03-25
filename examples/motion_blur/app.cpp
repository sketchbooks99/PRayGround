#include "app.h"

float easeOutExpo(float x)
{
    return x == 1.0f ? 1.0f : 1.0f - powf(2, -10.0f * x);
}

void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<Vec4u*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bitmap.devicePtr());

    CUDA_SYNC_CHECK();
}

// ----------------------------------------------------------------
void App::setup()
{
    pgSetFrameRate(30);

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
    Module module = pipeline.createModuleFromCudaFile(context, "kernels.cu");

    // レンダリング結果を保存する用のBitmapを用意
    result_bitmap.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());
    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());
    accum_bitmap.allocateDevicePtr();
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.frame = 0;
    params.samples_per_launch = 4;
    params.light =
    {
        .pos = Vec3f(0, 9.9f, 0), 
        .color = Vec3f(1.0f), 
        .intensity = 5.0f
    };
    params.result_buffer = reinterpret_cast<Vec4u*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bitmap.devicePtr());

    // カメラの設定
    camera.setOrigin(Vec3f(0.0f, 0.0f, 40.0f));
    camera.setLookat(Vec3f(0.0f, 0.0f, 0.0f));
    camera.setUp(Vec3f(0.0f, 1.0f, 0.0f));
    camera.setFov(40.0f);
    camera.setAspect((float)params.width / params.height);

    // Raygen プログラム用のデータ準備
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, module, "__raygen__pinhole");
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera = camera.getData();
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
    uint32_t constant_prg_id = setupCallable(module, "__direct_callable__constant", "");
    uint32_t checker_prg_id = setupCallable(module, "__direct_callable__checker", "");

    // 環境マッピング用のテクスチャとデータを準備
    auto env_color = make_shared<ConstantTexture>(Vec3f(0.5f), constant_prg_id);
    env_color->copyToDevice();
    auto env = EnvironmentEmitter{env_color};
    env.copyToDevice();

    // Miss プログラムの生成とSBTデータの用意
    ProgramGroup miss_prg = pipeline.createMissProgram(context, module, "__miss__envmap");
    MissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env.devicePtr();
    sbt.setMissRecord(miss_record);

    uint32_t sbt_offset = 0;
    uint32_t sbt_idx = 0;
    uint32_t instance_id = 0;

    // ShapeInstanceは1つのShapeのみ保持する前提でShapeInstanceをセットアップするLambda関数
    auto setupShapeInstance = [&](ProgramGroup hitgroup_prg, ShapeInstance& instance, shared_ptr<Texture> texture, bool is_update=false)
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
            .texture = texture->getData()
        };

        sbt.addHitgroupRecord(record);

        instance.allowCompaction();
        if (is_update)
            instance.allowUpdate();
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);
        instance.buildAccel(context, stream);

        instance_accel.addInstance(instance);

        instance_id++;
        sbt_offset += SBT::NRay;
        sbt_idx++;
    };

    auto plane_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__plane", "__intersection__plane");
    auto sphere_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__sphere", "__intersection__sphere");

    // Scene ==========================================================================
    // Cornel box用のテクスチャを用意
    auto floor_checker = make_shared<CheckerTexture>(Vec3f(0.9f), Vec3f(0.3f), 10, checker_prg_id);
    auto red = make_shared<ConstantTexture>(Vec3f(0.8f, 0.05f, 0.05f), constant_prg_id);
    auto green = make_shared<ConstantTexture>(Vec3f(0.05f, 0.8f, 0.05f), constant_prg_id);
    auto white = make_shared<ConstantTexture>(Vec3f(0.8f), constant_prg_id);

    // Cornel box
    {

        ShapeInstance floor {
            ShapeType::Custom, 
            make_shared<Plane>(Vec2f(-10.0f, -10.0f), Vec2f(10.0f, 10.0f)),
            Matrix4f::translate(Vec3f(0.0f, -10.0f, 0.0f))
        };

        ShapeInstance ceiling {
            ShapeType::Custom, 
            make_shared<Plane>(Vec2f(-10.0f, -10.0f), Vec2f(10.0f, 10.0f)),
            Matrix4f::translate(Vec3f(0.0f, 10.0f, 0.0f))
        };

        ShapeInstance back {
            ShapeType::Custom, 
            make_shared<Plane>(Vec2f(-10.0f, -10.0f), Vec2f(10.0f, 10.0f)),
            // x軸中心に90度回転
            Matrix4f::translate(Vec3f(0.0f, 0.0f, -10.0f)) * Matrix4f::rotate(math::pi / 2, {1.0f, 0.0f, 0.0f}) 
        };
        
        ShapeInstance right_wall {
            ShapeType::Custom, 
            make_shared<Plane>(Vec2f(-10.0f, -10.0f), Vec2f(10.0f, 10.0f)),
            // z軸中心に90度回転
            Matrix4f::translate(Vec3f(10.0f, 0.0f, 0.0f)) * Matrix4f::rotate(math::pi / 2, {0.0f, 0.0f, 1.0f})
        };

        ShapeInstance left_wall {
            ShapeType::Custom, 
            make_shared<Plane>(Vec2f(-10.0f, -10.0f), Vec2f(10.0f, 10.0f)),
            // z軸中心に90度回転
            Matrix4f::translate(Vec3f(-10.0f, 0.0f, 0.0f)) * Matrix4f::rotate(math::pi / 2, {0.0f, 0.0f, 1.0f})
        };

        setupShapeInstance(plane_prg, floor, floor_checker);
        setupShapeInstance(plane_prg, ceiling, white);
        setupShapeInstance(plane_prg, right_wall, red);
        setupShapeInstance(plane_prg, left_wall, green);
        setupShapeInstance(plane_prg, back, white);
    }

    // Sphere (with motion blur)
    {
        sphere_pos = sphere_prev_pos = Vec3f(0.0f, 2.5f, 0.0f);

        auto sphere = make_shared<Sphere>();
        sphere->setSbtIndex(sbt_idx);
        sphere->copyToDevice();

        HitgroupRecord record;
        sphere_prg.recordPackHeader(&record);
        record.data = 
        {
            .shape_data = sphere->devicePtr(),
            .texture = white->getData()
        };

        sbt.addHitgroupRecord(record);

        // 球体用のGASを用意
        GeometryAccel sphere_gas{ShapeType::Custom};
        sphere_gas.addShape(sphere);
        sphere_gas.allowCompaction();
        sphere_gas.allowUpdate();
        sphere_gas.build(context, stream);

        // Motion blur用の開始と終了点における変換行列を用意
        Matrix4f begin_matrix = Matrix4f::translate(sphere_prev_pos) * Matrix4f::scale(3.0f);
        Matrix4f end_matrix = Matrix4f::translate(sphere_pos) * Matrix4f::scale(3.0f);

        // Matrix motion 用のTransformを用意
        matrix_transform = Transform{TransformType::MatrixMotion};
        matrix_transform.setChildHandle(sphere_gas.handle());
        matrix_transform.setMotionOptions(instance_accel.motionOptions());
        matrix_transform.setMatrixMotionTransform(begin_matrix, end_matrix);
        matrix_transform.copyToDevice();
        // childHandleからTransformのTraversableHandleを生成
        matrix_transform.buildHandle(context);

        // Instanceの生成
        Instance instance;
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);
        instance.setTraversableHandle(matrix_transform.handle());

        instance_accel.addInstance(instance);
    }

    // Sphere (without motion blur)
    {
        sphere2 = ShapeInstance{
            ShapeType::Custom,
            make_shared<Sphere>(),
            Matrix4f::translate(sphere_pos - Vec3f(0.0f, 8.5f, 0.0f)) * Matrix4f::scale(3.0f)
        };
        setupShapeInstance(sphere_prg, sphere2, white, true);
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
    initResultBufferOnDevice();

    pgSetWindowName(toString(pgGetFrameRate()));

    float time = pgGetElapsedTimef();
    if (is_move) {
        float x = fmodf(time, 1.0f);
        sphere_prev_pos = sphere_pos;
        sphere_pos.x() = -5.0f + easeOutExpo(x) * 10.0f;

        Matrix4f begin_matrix = Matrix4f::translate(sphere_prev_pos) * Matrix4f::scale(3.0f);
        Matrix4f end_matrix = Matrix4f::translate(sphere_pos) * Matrix4f::scale(3.0f);

        matrix_transform.setMatrixMotionTransform(begin_matrix, end_matrix);
        matrix_transform.copyToDevice();

        sphere2.setTransform(Matrix4f::translate(sphere_pos - Vec3f(0.0f, 8.5f, 0.0f)) * Matrix4f::scale(3.0f));
    }

    instance_accel.update(context, stream);

    params.frame++;
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

// ----------------------------------------------------------------
void App::keyPressed(int key)
{
    if (key == Key::S)
    {
        is_move = !is_move;
    }
}
