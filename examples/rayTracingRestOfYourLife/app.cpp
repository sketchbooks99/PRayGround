#include "app.h"

void App::initResultBufferOnDevice()
{
    params.subframe_index = 0;

    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<float4*>(accum_bitmap.devicePtr());

    CUDA_SYNC_CHECK();
}

void App::handleCameraUpdate()
{
    if (!camera_update)
        return;
    camera_update = false;

    float3 U, V, W;
    camera.UVWFrame(U, V, W);

    RaygenRecord* rg_record = reinterpret_cast<RaygenRecord*>(sbt.raygenRecord());
    RaygenData rg_data;
    rg_data.camera =
    {
        .origin = camera.origin(),
        .lookat = camera.lookat(),
        .U = U, 
        .V = V, 
        .W = W,
        .farclip = camera.farClip()
    };

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(&rg_record->data),
        &rg_data, sizeof(RaygenData),
        cudaMemcpyHostToDevice
    ));

    initResultBufferOnDevice();
}

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
    raygen_module = pipeline.createModuleFromCudaFile(context, "cuda/raygen.cu");
    miss_module = pipeline.createModuleFromCudaFile(context, "cuda/miss.cu");
    hitgroups_module = pipeline.createModuleFromCudaFile(context, "cuda/hitgroups.cu");
    textures_module = pipeline.createModuleFromCudaFile(context, "cuda/textures.cu");
    surfaces_module = pipeline.createModuleFromCudaFile(context, "cuda/surfaces.cu");

    // レンダリング結果を保存する用のBitmapを用意
    result_bitmap.allocate(Bitmap::Format::RGBA, pgGetWidth(), pgGetHeight());
    accum_bitmap.allocate(FloatBitmap::Format::RGBA, pgGetWidth(), pgGetHeight());

    // LaunchParamsの設定
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.samples_per_launch = 1;
    params.max_depth = 5;
    params.white = 5.0f;

    initResultBufferOnDevice();

    // カメラの設定
    camera.setOrigin(make_float3(278.0f, 278.0f, -800.0f));
    camera.setLookat(make_float3(278.0f, 278.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFarClip(5000);
    camera.setFov(40.0f);
    float3 U, V, W;
    camera.UVWFrame(U, V, W);
    camera.enableTracking(pgGetCurrentWindow());

    // Raygenプログラム
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, raygen_module, "__raygen__pinhole");
    // Raygenプログラム用のShader Binding Tableデータ
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera =
    {
        .origin = camera.origin(),
        .lookat = camera.lookat(),
        .U = U,
        .V = V,
        .W = W,
        .farclip = camera.farClip()
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
    uint32_t constant_prg_id = setupCallable(textures_module, DC_FUNC_STR("constant"), "");

    // Surface用のCallableプログラム 
    // Diffuse
    uint32_t diffuse_sample_bsdf_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("sample_diffuse"), CC_FUNC_STR("bsdf_diffuse"));
    uint32_t diffuse_pdf_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("pdf_diffuse"), "");
    // Dielectric
    uint32_t dielectric_sample_bsdf_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("sample_dielectric"), CC_FUNC_STR("bsdf_dielectric"));
    uint32_t dielectric_pdf_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("pdf_dielectric"), "");
    // AreaEmitter
    uint32_t area_emitter_prg_id = setupCallable(surfaces_module, DC_FUNC_STR("area_emitter"), "");

    // Shape用のCallableプログラム(主に面光源サンプリング用)
    uint32_t plane_sample_pdf_prg_id = setupCallable(hitgroups_module, DC_FUNC_STR("rnd_sample_plane"), CC_FUNC_STR("pdf_plane"));

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
    sbt.setMissRecord(miss_record);

    // Hitgroupプログラム
    // Plane
    auto plane_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("plane"), IS_FUNC_STR("plane"));
    // Sphere
    auto sphere_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("sphere"), IS_FUNC_STR("sphere"));
    // Triangle mesh
    auto mesh_prg = pipeline.createHitgroupProgram(context, hitgroups_module, CH_FUNC_STR("mesh"));

    struct Primitive
    {
        shared_ptr<Shape> shape;
        shared_ptr<Material> material;
        uint32_t sample_bsdf_id;
        uint32_t pdf_id;
    };

    uint32_t sbt_idx = 0;
    uint32_t sbt_offset = 0;
    uint32_t instance_id = 0;
    // ShapeとMaterialのデータをGPU上に準備しHitgroup用のSBTデータを追加するLambda関数
    auto setupPrimitive = [&](ProgramGroup& prg, const Primitive& primitive, const Matrix4f& transform)
    {
        // データをGPU側に用意
        primitive.shape->copyToDevice();
        primitive.shape->setSbtIndex(sbt_idx);
        primitive.material->copyToDevice();

        // Shader Binding Table へのデータの登録
        HitgroupRecord record;
        prg.recordPackHeader(&record);
        record.data = 
        {
            .shape_data = primitive.shape->devicePtr(), 
            .surface_info = 
            {
                .data = primitive.material->devicePtr(),
                .sample_id = primitive.sample_bsdf_id,
                .bsdf_id = primitive.sample_bsdf_id,
                .pdf_id = primitive.pdf_id,
                .type = primitive.material->surfaceType()
            }
        };

        sbt.addHitgroupRecord(record);
        sbt_idx++;

        // GASをビルドし、IASに追加
        ShapeInstance instance{primitive.shape->type(), primitive.shape, transform};
        instance.allowCompaction();
        instance.buildAccel(context, stream);
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);

        scene_ias.addInstance(instance);

        instance_id++;
        sbt_offset += RtRestLifeSBT::NRay;
    };

    std::vector<AreaEmitterInfo> area_emitter_infos;
    // 面光源用のSBTデータを用意しグローバル情報としての光源情報を追加するLambda関数
    // 光源サンプリング時にCallable関数ではOptixInstanceに紐づいた行列情報を取得できないので
    // 行列情報をAreaEmitterInfoに一緒に設定しておく
    // ついでにShapeInstanceによって光源用のGASも追加
    auto setupAreaEmitter = [&](
        ProgramGroup& prg,
        shared_ptr<Shape> shape,
        AreaEmitter area, Matrix4f transform, 
        uint32_t sample_pdf_id
    )
    {
        // Plane or Sphereにキャスト可能かチェック
        ASSERT(dynamic_pointer_cast<Plane>(shape) || dynamic_pointer_cast<Sphere>(shape), "The shape of area emitter must be a plane or sphere.");
        
        shape->copyToDevice();
        shape->setSbtIndex(sbt_idx);
        area.copyToDevice();

        HitgroupRecord record;
        prg.recordPackHeader(&record);
        record.data = 
        {
            .shape_data = shape->devicePtr(), 
            .surface_info = 
            {
                .data = area.devicePtr(),
                .sample_id = sample_pdf_id,
                .bsdf_id = area_emitter_prg_id,
                .pdf_id = sample_pdf_id,
                .type = SurfaceType::AreaEmitter
            }
        };

        sbt_idx++;

        sbt.addHitgroupRecord(record);

        // GASをビルドし、IASに追加
        ShapeInstance instance{shape->type(), shape, transform};
        instance.allowCompaction();
        instance.buildAccel(context, stream);
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);

        scene_ias.addInstance(instance);

        instance_id++;
        sbt_offset += RtRestLifeSBT::NRay;

        AreaEmitterInfo area_emitter_info = 
        {
            .shape_data = shape->devicePtr(), 
            .objToWorld = transform,
            .worldToObj = transform.inverse(), 
            .sample_id = sample_pdf_id, 
            .pdf_id = sample_pdf_id,
            .gas_handle = instance.handle()
        };
        area_emitter_infos.push_back(area_emitter_info);
    };

    auto green = make_shared<ConstantTexture>(make_float3(0.05f, 0.8f, 0.05f), constant_prg_id);
    auto red = make_shared<ConstantTexture>(make_float3(0.8f, 0.05f, 0.05f), constant_prg_id);
    auto white = make_shared<ConstantTexture>(make_float3(0.8f, 0.8f, 0.8f), constant_prg_id);

    // Ceiling
    {
        auto plane = make_shared<Plane>(make_float2(0.0f,0.0f), make_float2(555.0f,555.0f));
        auto diffuse = make_shared<Diffuse>(white);
        auto transform = Matrix4f::translate({0.0f, 555.0f, 0.0f});
        Primitive ceiling{plane, diffuse, diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id};
        setupPrimitive(plane_prg, ceiling, transform);
    }

    // Red wall
    {
        auto plane = make_shared<Plane>(make_float2(0.0f,0.0f), make_float2(555.0f,555.0f));
        auto diffuse = make_shared<Diffuse>(red);
        auto transform = Matrix4f::rotate(math::pi / 2.0f, {0.0f, 0.0f, 1.0f});
        Primitive left_wall{plane, diffuse, diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id};
        setupPrimitive(plane_prg, left_wall, transform);
    }

    // Green wall
    {
        auto plane = make_shared<Plane>(make_float2(0.0f,0.0f), make_float2(555.0f,555.0f));
        auto diffuse = make_shared<Diffuse>(green);
        auto transform = Matrix4f::translate({555.0f, 0.0f, 0.0f}) * Matrix4f::rotate(math::pi / 2.0f, {0.0f, 0.0f, 1.0f});
        Primitive right_wall{plane, diffuse, diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id};
        setupPrimitive(plane_prg, right_wall, transform);
    }

    // Back
    {
        auto plane = make_shared<Plane>(make_float2(0.0f,0.0f), make_float2(555.0f,555.0f));
        auto diffuse = make_shared<Diffuse>(white);
        auto transform = Matrix4f::translate({0.0f, 555.0f, 555.0f}) * Matrix4f::rotate(math::pi / 2.0f, {1.0f, 0.0f, 0.0f});
        Primitive back{plane, diffuse, diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id};
        setupPrimitive(plane_prg, back, transform);
    }

    // Floor
    {
        auto plane = make_shared<Plane>(make_float2(0.0f,0.0f), make_float2(555.0f,555.0f));
        auto diffuse = make_shared<Diffuse>(white);
        auto transform = Matrix4f::translate({0.0f, 0.0f, 0.0f});
        Primitive floor{plane, diffuse, diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id};
        setupPrimitive(plane_prg, floor, transform);
    }

    // Cube
    {
        auto mesh = make_shared<TriangleMesh>("resources/model/cube.obj");
        auto diffuse = make_shared<Diffuse>(white);
        auto transform = Matrix4f::translate({347.5f, 165, 377.5f}) * Matrix4f::rotate(math::radians(15.0f), {0.0f, 1.0f, 0.0f}) * Matrix4f::scale({82.5f, 165.0f, 82.5f});
        Primitive cube{mesh, diffuse, diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id};
        setupPrimitive(mesh_prg, cube, transform);
    }

    // Glass sphere
    {
        auto sphere = make_shared<Sphere>(make_float3(190.0f, 90.0f, 190.0f), 90.0f);
        auto glass = make_shared<Dielectric>(white, 1.5f);
        auto transform = Matrix4f::identity();
        Primitive glass_sphere{sphere, glass, dielectric_sample_bsdf_prg_id, dielectric_pdf_prg_id};
        setupPrimitive(sphere_prg, glass_sphere, transform);
    }

    // Ceiling light
    {
        // Shape
        auto plane_light = make_shared<Plane>(make_float2(213.0f, 227.0f), make_float2(343.0f, 332.0f));
        // Texture
        auto light_white = make_shared<ConstantTexture>(make_float3(1.0f), constant_prg_id);
        light_white->copyToDevice();
        // Area emitter
        auto plane_area_emitter = AreaEmitter(light_white, 15.0f);
        Matrix4f transform = Matrix4f::translate({0.0f, 554.0f, 0.0f});
        setupAreaEmitter(plane_prg, plane_light, plane_area_emitter, transform, plane_sample_pdf_prg_id);
    }

    // 光源データをGPU側にコピー
    CUDABuffer<AreaEmitterInfo> d_area_emitter_infos;
    d_area_emitter_infos.copyToDevice(area_emitter_infos);
    params.lights = d_area_emitter_infos.deviceData();
    params.num_lights = static_cast<int>(area_emitter_infos.size());

    CUDA_CHECK(cudaStreamCreate(&stream));
    scene_ias.build(context, stream);
    sbt.createOnDevice();
    params.handle = scene_ias.handle();
    pipeline.create(context);
    d_params.allocate(sizeof(LaunchParams));

    // GUI setting
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    const char* glsl_version = "#version 150";
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(pgGetCurrentWindow()->windowPtr(), true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

// ----------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();

    params.subframe_index++;
    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

    // OptiX レイトレーシングカーネルの起動
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

    // レンダリング結果をデバイスから取ってくる
    result_bitmap.copyFromDevice();
}

// ----------------------------------------------------------------
void App::draw()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Ray Tracing Rest of Your Life");

    ImGui::SliderFloat("White", &params.white, 1.0f, 1000.0f);
    ImGui::Text("Camera info:");
    ImGui::Text("Origin: %f %f %f", camera.origin().x, camera.origin().y, camera.origin().z);
    ImGui::Text("Lookat: %f %f %f", camera.lookat().x, camera.lookat().y, camera.lookat().z);
    ImGui::Text("Up: %f %f %f", camera.up().x, camera.up().y, camera.up().z);

    float farclip = camera.farClip();
    ImGui::SliderFloat("far clip", &farclip, 500.0f, 10000.0f);
    if (farclip != camera.farClip()) {
        camera.setFarClip(farclip);
        camera_update = true;
    }

    ImGui::Text("Frame rate: %.3f ms/frame (%.2f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Subframe index: %d", params.subframe_index);

    ImGui::End();

    ImGui::Render();

    result_bitmap.draw(0, 0);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // if (params.subframe_index == 4096)
    //     result_bitmap.write(pathJoin(pgAppDir(), "rtRestOfYourLife.jpg"));
}

// ----------------------------------------------------------------
void App::close()
{
    env.free();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

// ----------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    camera_update = true;
}

// ----------------------------------------------------------------
void App::mouseScrolled(float xoffset, float yoffset)
{
    camera_update = true;
}