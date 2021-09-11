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

    ias = InstanceAccel{ InstanceAccel::Type::Instances };
    ias.allowUpdate();

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
    accum_bitmap.allocate(FloatBitmap::Format::RGBA, pgGetWidth(), pgGetHeight());
    accum_bitmap.allocateDevicePtr();
    normal_bitmap.allocate(FloatBitmap::Format::RGB, pgGetWidth(), pgGetHeight());
    normal_bitmap.allocateDevicePtr();
    albedo_bitmap.allocate(FloatBitmap::Format::RGB, pgGetWidth(), pgGetHeight());
    albedo_bitmap.allocateDevicePtr();
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;
    params.subframe_index = 0;
    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<float4*>(accum_bitmap.devicePtr());
    params.normal_buffer = reinterpret_cast<float3*>(normal_bitmap.devicePtr());
    params.albedo_buffer = reinterpret_cast<float3*>(albedo_bitmap.devicePtr());

    // カメラの設定
    camera.setOrigin(make_float3(0.0f, 0.0f, 40.0f));
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

    EmptyRecord callable_record = {};

    // 単色テクスチャ用のプログラム
    auto [constant_prg, constant_prg_id] = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__constant", "");
    constant_prg.recordPackHeader(&callable_record);
    sbt.addCallablesRecord(callable_record);
    // チェッカーボード用のプログラム
    callable_record = {};
    auto [checker_prg, checker_prg_id] = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__checker", "");
    checker_prg.recordPackHeader(&callable_record);
    sbt.addCallablesRecord(callable_record);

    // 環境マッピング用のテクスチャとデータを準備
    auto env_color = make_shared<ConstantTexture>(make_float3(0.5f), constant_prg_id);
    env_color->copyToDevice();
    auto env_emitter = make_shared<EnvironmentEmitter>(env_color);
    env_emitter->copyToDevice();

    // Miss プログラムの生成とSBTデータの用意
    ProgramGroup miss_prg = pipeline.createMissProgram(context, miss_module, "__miss__envmap");
    MissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env_emitter->devicePtr();
    sbt.setMissRecord(miss_record, miss_record);

    // Cornel box用のテクスチャを用意
    auto floor_checker = make_shared<CheckerTexture>(make_float3(0.9f), make_float3(0.3f), 10, checker_prg_id);
    auto red = make_shared<ConstantTexture>(make_float3(0.8f, 0.05f, 0.05f), constant_prg_id);
    auto green = make_shared<ConstantTexture>(make_float3(0.05f, 0.8f, 0.05f), constant_prg_id);
    auto white = make_shared<ConstantTexture>(make_float3(0.8f), constant_prg_id);
    // Cornel box用のマテリアルを用意
    auto floor_diffuse = make_shared<Diffuse>(floor_checker);
    auto red_diffuse = make_shared<Diffuse>(red);
    auto green_diffuse = make_shared<Diffuse>(green);
    auto white_diffuse = make_shared<Diffuse>(white);
    floor_diffuse->copyToDevice(); // GPU上にデータをコピー
    red_diffuse->copyToDevice();
    green_diffuse->copyToDevice();
    white_diffuse->copyToDevice();

    // Diffuseマテリアル用のCallableプログラムを生成
    // Diffuseではシャドウレイ用にoptixTrace()を呼び出すため、
    // Direct callableではなくContinuation callableにする
    auto [diffuse_prg, diffuse_prg_id] = pipeline.createCallablesProgram(context, surfaces_module, "", "__continuation_callable__diffuse");
    callable_record = {};
    diffuse_prg.recordPackHeader(&callable_record);
    sbt.addCallablesRecord(callable_record);

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
        Matrix4f::translate(make_float3(0.0f, 0.0f, -10.0f)) * Matrix4f::rotate(constants::pi / 2, {1.0f, 0.0f, 0.0f}) 
    };

    ShapeInstance right_wall {
        ShapeType::Custom, 
        make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f)),
        // z軸中心に90度回転
        Matrix4f::translate(make_float3(10.0f, 0.0f, 0.0f)) * Matrix4f::rotate(constants::pi / 2, {0.0f, 0.0f, 1.0f})
    };

    ShapeInstance left_wall {
        ShapeType::Custom, 
        make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f)),
        // z軸中心に90度回転
        Matrix4f::translate(make_float3(-10.0f, 0.0f, 0.0f)) * Matrix4f::rotate(constants::pi / 2, {0.0f, 0.0f, 1.0f})
    };

    cornel_box.push_back({floor,         floor_diffuse});
    cornel_box.push_back({ceiling,       white_diffuse});
    cornel_box.push_back({back,          white_diffuse});
    cornel_box.push_back({right_wall,    red_diffuse});
    cornel_box.push_back({left_wall,     green_diffuse});

    uint32_t sbt_offset = 0;
    uint32_t sbt_idx = 0;
    uint32_t instance_id = 0;

    // Plane用のHitgroupプログラムの生成
    auto plane_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__plane", "__intersection__plane");
    auto plane_shadow_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__shadow", "__intersection__plane");

    for (auto& [plane_instance, material] : cornel_box)
    {
        for (auto& plane : plane_instance.shapes()) {
            // Hitgroupプログラム用のデータを準備
            plane->copyToDevice();    
            plane->setSbtIndex(sbt_idx);

            // SBT用のHitgroupDataの生成
            HitgroupRecord record;
            plane_prg.recordPackHeader(&record);
            record.data =
            {
                .shape_data = plane->devicePtr(),
                .surface_data = material->devicePtr(),
                .surface_program_id = diffuse_prg_id,
                .surface_type = SurfaceType::Material
            };

            // シャドウレイ用のHitgroupData
            // 衝突したかどうかをPayloadに設定するだけなので、データは設定しない
            HitgroupRecord shadow_record;
            plane_shadow_prg.recordPackHeader(&shadow_record);

            sbt.addHitgroupRecord(record, shadow_record);
        }
        // Plane用のGASとインスタンスを作成
        plane_instance.allowCompaction();
        plane_instance.buildAccel(context, stream);
        //plane_instance.setSBTOffset(0);
        plane_instance.setSBTOffset(sbt_offset);
        plane_instance.setId(instance_id);

        // InstanceをInstanceAccelに追加
        ias.addInstance(plane_instance);

        instance_id++;
        sbt_offset += DynamicUpdateSBT::NRay * plane_instance.shapes().size();
        sbt_idx++;
    }

    auto [area_prg, area_prg_id] = pipeline.createCallablesProgram(context, surfaces_module, "__direct_callable__area_emitter", "");
    callable_record = {};
    area_prg.recordPackHeader(&callable_record);
    sbt.addCallablesRecord(callable_record);

    // 天井の面光源
    {
        // 天井の面光源用のテクスチャの準備
        ceiling_texture = make_shared<ConstantTexture>(make_float3(1.0f), constant_prg_id);
        ceiling_texture->copyToDevice(); // GPU上にデータをコピー
        // 天井の光源のデータを準備
        ceiling_emitter = make_shared<AreaEmitter>(ceiling_texture, 10.0f);
        ceiling_emitter->copyToDevice();   // GPU上にデータをコピー

        ShapeInstance ceiling_light {
            ShapeType::Custom, 
            make_shared<Plane>(make_float2(-2.5f, -2.5f), make_float2(2.5f, 2.5f)), 
            Matrix4f::translate(make_float3(0.0f, 9.9f, 0.0f))
        };
        ceiling_light.shapes()[0]->copyToDevice();
        ceiling_light.shapes()[0]->setSbtIndex(sbt_idx);
        ceiling_light.allowCompaction();
        ceiling_light.buildAccel(context, stream);
        ceiling_light.setSBTOffset(sbt_offset);
        ceiling_light.setId(instance_id);
        ias.addInstance(ceiling_light);
        sbt_offset += DynamicUpdateSBT::NRay;
        sbt_idx++;
        instance_id++;

        // 天井の光源用のHitgroupDataの生成
        HitgroupRecord record;
        plane_prg.recordPackHeader(&record);
        record.data =
        {
            .shape_data = ceiling_light.shapes()[0]->devicePtr(),
            .surface_data = ceiling_emitter->devicePtr(), 
            .surface_program_id = area_prg_id,
            .surface_type = SurfaceType::AreaEmitter
        };

        HitgroupRecord shadow_record;
        plane_shadow_prg.recordPackHeader(&shadow_record);
        sbt.addHitgroupRecord(record, shadow_record);
    }

    // Sphere用のHitgroupプログラムの生成
    auto sphere_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__sphere", "__intersection__sphere");
    auto sphere_shadow_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__shadow", "__intersection__sphere");

    // 金属&誘電体マテリアル用
    auto [conductor_prg, conductor_prg_id] = pipeline.createCallablesProgram(context, surfaces_module, "", "__continuation_callable__conductor");
    callable_record = {};
    conductor_prg.recordPackHeader(&callable_record);
    sbt.addCallablesRecord(callable_record);
    auto [glass_prg, glass_prg_id] = pipeline.createCallablesProgram(context, surfaces_module, "", "__continuation_callable__dielectric");
    callable_record = {};
    glass_prg.recordPackHeader(&callable_record);
    sbt.addCallablesRecord(callable_record);

    // Sphere1
    {
        // Sphere1用の金属マテリアル
        auto metal_texture = make_shared<CheckerTexture>(make_float3(0.8f, 0.8f, 0.05f), make_float3(0.3f), 10, checker_prg_id);
        metal_texture->copyToDevice();
        auto metal = make_shared<Conductor>(metal_texture);
        metal->copyToDevice();
        sphere1_pos = make_float3(0.0f);
        sphere1 = ShapeInstance(
            ShapeType::Custom, 
            make_shared<Sphere>(make_float3(0.0f), 1.0f), 
            Matrix4f::translate(sphere1_pos) * Matrix4f::scale(2.5f)
        );
        sphere1.shapes()[0]->copyToDevice();
        sphere1.shapes()[0]->setSbtIndex(sbt_idx);
        sphere1.allowCompaction();
        sphere1.buildAccel(context, stream);
        sphere1.setSBTOffset(sbt_offset);
        sphere1.setId(instance_id);
        ias.addInstance(sphere1);
        sbt_offset += DynamicUpdateSBT::NRay;
        sbt_idx++;
        instance_id++;

        // Sphere1用のHitgroupDataの生成
        HitgroupRecord sphere1_record;
        sphere_prg.recordPackHeader(&sphere1_record);
        sphere1_record.data = 
        {
            .shape_data = sphere1.shapes()[0]->devicePtr(),
            .surface_data = metal->devicePtr(),
            .surface_program_id = conductor_prg_id,
            .surface_type = SurfaceType::Material
        };

        HitgroupRecord sphere1_shadow_record;
        sphere_shadow_prg.recordPackHeader(&sphere1_shadow_record);

        sbt.addHitgroupRecord(sphere1_record, sphere1_shadow_record);
    }

    // Sphere2
    {
        // Sphere2用の誘電体マテリアル
        auto glass_texture = make_shared<ConstantTexture>(make_float3(1.0f), constant_prg_id);
        glass_texture->copyToDevice();
        auto glass = make_shared<Dielectric>(glass_texture, 1.5f);
        glass->copyToDevice();
        sphere2_pos = make_float3(0.0f);
        sphere2 = ShapeInstance(
            ShapeType::Custom, 
            make_shared<Sphere>(make_float3(0.0f), 1.0f), 
            Matrix4f::translate(sphere2_pos) * Matrix4f::scale(2.5f)
        );
        sphere2.shapes()[0]->copyToDevice();
        sphere2.shapes()[0]->setSbtIndex(sbt_idx);
        sphere2.allowCompaction();
        sphere2.buildAccel(context, stream);
        sphere2.setSBTOffset(sbt_offset);
        sphere2.setId(instance_id);
        ias.addInstance(sphere2);

        // Sphere1用のHitgroupDataの生成
        HitgroupRecord sphere2_record;
        sphere_prg.recordPackHeader(&sphere2_record);
        sphere2_record.data = 
        {
            .shape_data = sphere2.shapes()[0]->devicePtr(),
            .surface_data = glass->devicePtr(),
            .surface_program_id = glass_prg_id,
            .surface_type = SurfaceType::Material
        };

        HitgroupRecord sphere2_shadow_record;
        sphere_shadow_prg.recordPackHeader(&sphere2_shadow_record);

        sbt.addHitgroupRecord(sphere2_record, sphere2_shadow_record);
    }
    
    // Shader Binding Table のデータをGPU上に生成
    sbt.createOnDevice();
    // レイトレーシング用のTraversableHandle を設定
    ias.build(context, stream);
    params.handle = ias.handle();
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
    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocateDevicePtr();
    normal_bitmap.allocateDevicePtr();
    albedo_bitmap.allocateDevicePtr();
    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<float4*>(accum_bitmap.devicePtr());
    params.normal_buffer = reinterpret_cast<float3*>(normal_bitmap.devicePtr());
    params.albedo_buffer = reinterpret_cast<float3*>(albedo_bitmap.devicePtr());

    float time = pgGetElapsedTime<float>();

    sphere1_pos = 5.0f * make_float3(sinf(time) * sinf(time / 3.0f), cosf(time), sinf(time) * cosf(time / 3.0f));
    sphere2_pos = 5.0f * make_float3(sinf(-time) * sinf(time / 3.0f), cosf(-time), sinf(-time) * cosf(time / 3.0f));
    sphere1.setTransform(Matrix4f::translate(sphere1_pos) * Matrix4f::scale(2.5f + 1.25 * sinf(time)));
    sphere2.setTransform(Matrix4f::translate(sphere2_pos) * Matrix4f::scale(2.5f + 1.25 * cosf(time)));

    ias.update(context, stream);

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
    normal_bitmap.copyFromDevice();
    albedo_bitmap.copyFromDevice();
}

// ----------------------------------------------------------------
void App::draw()
{
    result_bitmap.draw(0, 0, pgGetWidth() / 2, pgGetHeight() / 2);
    normal_bitmap.draw(pgGetWidth() / 2, 0, pgGetWidth() / 2, pgGetHeight() / 2);
    albedo_bitmap.draw(0, pgGetHeight() / 2, pgGetWidth() / 2, pgGetHeight() / 2);
}

// ----------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    /// @todo カメラの位置変更
}
