#include "app.h"

// ----------------------------------------------------------------
void App::setup()
{
    // CUDAの初期化とOptixDeviceContextの生成
    stream = 0;
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    context.create();

    // Instance acceleration structureの初期化
    scene_ias = InstanceAccel{InstanceAccel::Type::Instances};

    // パイプラインの設定
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(5);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);

    // OptixModuleをCUDAファイルから生成
    Module raygen_module, miss_module, hitgroups_module, textures_module, surfaces_module;
    raygen_module = pipeline.createModuleFromCudaFile(context, "raygen.cu");
    miss_module = pipeline.createModuleFromCudaFile(context, "miss.cu");
    hitgroups_module = pipeline.createModuleFromCudaFile(context, "hitgroups.cu");
    textures_module = pipeline.createModuleFromCudaFile(context, "textures.cu");
    surfaces_module = pipeline.createModuleFromCudaFile(context, "surfaces.cu");

    // レンダリング結果を保存する用のBitmapを用意
    result_bitmap.allocate(Bitmap::Format::RGBA, pgGetWidth(), pgGetHeight());
    result_bitmap.allocateDevicePtr(); // GPU側のバッファを確保
    accum_bitmap.allocate(FloatBitmap::Format::RGBA, pgGetWidth(), pgGetHeight());
    accum_bitmap.allocateDevicePtr(); // GPU側のバッファを確保

    // LaunchParamsの設定
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;
    params.subframe_index = 0;
    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<float4*>(accum_bitmap.devicePtr());

    // カメラの設定
    camera.setOrigin(make_float3(20.0f, 10.0f, 30.0f));
    camera.setLookat(make_float3(0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFov(40.0f);

    // Raygenプログラム
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, raygen_module, "__raygen__pinhole");
    // Raygenプログラム用のShader Binding Tableデータ
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera = 
    {
        .origin = camera.origin(),
        .lookat = camera.lookat(),
        .up = camera.up(),
        .fov = camera.fov(), 
        .aspect = static_cast<float>(result_bitmap.width()) / result_bitmap.height()
    };
    sbt.setRaygenRecord(raygen_record);

    // Callable関数とShader Binding TableにCallable関数用のデータを登録するLambda関数
    auto prepareCallable = [&](const Module& module, const std::string& dc, const std::string& cc)
    {
        EmptyRecord callable_record = {};
        auto [prg, id] = pipeline.createCallablesProgram(context, module, dc, cc);
        prg.recordPackHeader(&callable_record);
        sbt.addCallablesRecord(callable_record);
        return pair<ProgramGroup, uint32_t>{prg, id};
    };

    // テクスチャ用のCallableプログラム
    auto [contant_prg, constant_prg_id] = prepareCallable(textures_module, DC_FUNC_STR("constant"), "");
    auto [checker_prg, checker_prg_id] = prepareCallable(textures_module, DC_FUNC_STR("checker"), "");
    auto [bitmap_prg, bitmap_prg_id] = prepareCallable(textures_module, DC_FUNC_STR("bitmap"), "");

    // Surface用のCallableプログラム 
    // Diffuse
    auto [diffuse_sample_bsdf_prg, diffuse_sample_bsdf_prg_id] = prepareCallable(surfaces_module, DC_FUNC_STR("sample_diffuse"), CC_FUNC_STR("bsdf_diffuse"));
    auto [diffuse_pdf_prg, diffuse_pdf_prg_id] = prepareCallable(surfaces_module, DC_FUNC_STR("pdf_diffuse"), "");
    // Conductor
    auto [conductor_sample_bsdf_prg, conductor_sample_bsdf_prg_id] = prepareCallable(surfaces_module, DC_FUNC_STR("sample_conductor"), CC_FUNC_STR("bsdf_conductor"));
    auto [conductor_pdf_prg, conductor_pdf_prg_id] = prepareCallable(surfaces_module, DC_FUNC_STR("pdf_conductor"), "");
    // Dielectric
    auto [dielectric_sample_bsdf_prg, dielectric_sample_bsdf_prg_id] = prepareCallable(surfaces_module, DC_FUNC_STR("sample_dielectric"), CC_FUNC_STR("bsdf_dielectric"));
    auto [dielectric_pdf_prg, dielectric_pdf_prg_id] = prepareCallable(surfaces_module, DC_FUNC_STR("pdf_dielectric"), "");
    // Disney
    auto [disney_sample_bsdf_prg, disney_sample_bsdf_prg_id] = prepareCallable(surfaces_module, DC_FUNC_STR("sample_disney"), CC_FUNC_STR("bsdf_disney"));
    auto [disney_pdf_prg, disney_pdf_prg_id] = prepareCallable(surfaces_module, DC_FUNC_STR("pdf_disney"), "");
    // AreaEmitter
    auto [area_emitter_prg, area_emitter_prg_id] = prepareCallable(surfaces_module, DC_FUNC_STR("area_emitter"), "");

    // Shape用のCallableプログラム(主に面光源サンプリング用)
    auto [sphere_sample_pdf_prg, sphere_sample_pdf_prg_id] = prepareCallable(hitgroups_module, DC_FUNC_STR("rnd_sample_sphere"), CC_FUNC_STR("pdf_sphere"));
    auto [plane_sample_pdf_prg, plane_sample_pdf_prg_id] = prepareCallable(hitgroups_module, DC_FUNC_STR("rnd_sample_plane"), CC_FUNC_STR("pdf_plane"));
    auto [triangle_sample_pdf_prg, triangle_sample_pdf_prg_id] = prepareCallable(hitgroups_module, DC_FUNC_STR("rnd_sample_triangle"), CC_FUNC_STR("pdf_triangle"));

    // 環境マッピング (Sphere mapping) 用のテクスチャとデータ準備
    auto env_texture = make_shared<ConstantTexture>(make_float3(0.0f), constant_prg_id);
    env_texture->copyToDevice();
    env = EnvironmentEmitter{env_texture};
    env.copyToDevice();

    // Missプログラム
    ProgramGroup miss_prg = pipeline.createMissProgram(context, miss_module, MS_FUNC_STR("envmap"));
    // Missプログラム用のShader Binding Tableデータ
    MissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env.devicePtr();
    miss_record = {};
    miss_prg.recordPackHeader(&miss_record);
    sbt.setMissRecord(miss_record, miss_record);

    // Hitgroupプログラム
    // Plane
    auto plane_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("plane"), IS_FUNC_STR("plane"));
    auto plane_shadow_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("shadow"), IS_FUNC_STR("plane"));
    // Sphere
    auto sphere_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("sphere"), IS_FUNC_STR("sphere"));
    auto sphere_shadow_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("shadow"), IS_FUNC_STR("sphere"));
    // Triangle mesh
    auto mesh_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("plane"));
    auto mesh_shadow_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("shadow"));

    uint32_t sbt_idx = 0;
    auto preparePrimitive = [&](ProgramGroup& prg, ProgramGroup& shadow_prg, const Primitive& primitive)
    {
        primitive.shape->copyToDevice();
        primitive.shape->setSbtIndex(sbt_idx);
        primitive.material->copyToDevice();

        HitgroupRecord record;
        prg.recordPackHeader(&record);
        record.data = 
        {
            .shape_data = primitive.shape->devicePtr(), 
            .surf_info = 
            {
                .data = primitive.material->devicePtr(),
                .sample_id = primitive.sample_bsdf_id,
                .bsdf_id = primitive.sample_bsdf_id,
                .pdf_id = primitive.pdf_id,
                .type = primitive.material->surfaceType()
            }
        };

        HitgroupRecord shadow_record;
        shadow_prg.recordPackHeader(&shadow_record);

        sbt.addHitgroupRecord(record, shadow_record);
        sbt_idx++;
    };
    
    std::vector<AreaEmitterInfo> area_infos;
    auto prepareAreaEmitter = [&](ProgramGroup& prg, ProgramGroup& shadow_prg, ShapeInstance shape_instance, AreaEmitter area)
    {
        
    };

    uint32_t sbt_offset = 0;
    uint32_t instance_id = 0;
    auto prepareShapeInstance = [&](ShapeInstance& shape_instance)
    {
        shape_instance.allowCompaction();
        shape_instance.buildAccel(context, stream);
        shape_instance.setSBTOffset(sbt_offset);
        shape_instance.setId(instance_id);

        scene_ias.addInstance(shape_instance);

        instance_id++;
        sbt_offset += PathTracingSBT::NRay * shape_instance.shapes().size();
    };
}

// ----------------------------------------------------------------
void App::update()
{
    
}

// ----------------------------------------------------------------
void App::draw()
{
}

// ----------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    float deltaX = x - pgGetPreviousMousePosition().x;
    float deltaY = y - pgGetPreviousMousePosition().y;
    float cam_length = length(camera.origin());
    float3 cam_dir = normalize(camera.origin() - camera.lookat());

    float theta = acosf(cam_dir.y);
    float phi = atan2(cam_dir.z, cam_dir.x);

    theta = min(math::pi - 0.01f, max(0.01f, theta - math::radians(deltaY * 0.25f)));
    phi += math::radians(deltaX * 0.25f);

    float cam_x = cam_length * sinf(theta) * cosf(phi);
    float cam_y = cam_length * cosf(theta);
    float cam_z = cam_length * sinf(theta) * sinf(phi);

    camera.setOrigin({ cam_x, cam_y, cam_z });

    RaygenRecord* rg_record = reinterpret_cast<RaygenRecord*>(sbt.raygenRecord());
    RaygenData rg_data;
    rg_data.camera =
    {
        .origin = camera.origin(),
        .lookat = camera.lookat(),
        .up = camera.up(),
        .fov = camera.fov(),
        .aspect = 1.0f
    };

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(&rg_record->data),
        &rg_data, sizeof(RaygenData),
        cudaMemcpyHostToDevice
    ));

    CUDA_SYNC_CHECK();

    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocateDevicePtr();
    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<float4*>(accum_bitmap.devicePtr());
    params.subframe_index = 0;

    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);
}

// ----------------------------------------------------------------
void App::mouseScrolled(float xoffset, float yoffset)
{
    float zoom = yoffset < 0 ? 1.1f : 1.0f / 1.1f;
    camera.setOrigin(camera.lookat() + (camera.origin() - camera.lookat()) * zoom);

    RaygenRecord* rg_record = reinterpret_cast<RaygenRecord*>(sbt.raygenRecord());
    RaygenData rg_data;
    rg_data.camera =
    {
        .origin = camera.origin(),
        .lookat = camera.lookat(),
        .up = camera.up(),
        .fov = camera.fov(),
        .aspect = 1.0f
    };

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(&rg_record->data),
        &rg_data, sizeof(RaygenData),
        cudaMemcpyHostToDevice
    ));

    CUDA_SYNC_CHECK();

    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocateDevicePtr();
    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<float4*>(accum_bitmap.devicePtr());
    params.subframe_index = 0;

    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);
}