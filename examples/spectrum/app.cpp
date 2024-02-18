#include "app.h"

void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<Vec4u*>(result_bitmap.deviceData());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bitmap.deviceData());

    CUDA_SYNC_CHECK();
}

void App::handleCameraUpdate()
{
    if (!camera_update)
        return;
    camera_update = false;

    RaygenRecord* rg_record = reinterpret_cast<RaygenRecord*>(sbt.deviceRaygenRecordPtr());
    RaygenData rg_data;
    rg_data.camera = camera.getData();

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
    Module module = pipeline.createModuleFromCudaFile(context, "kernels.cu");

    // レンダリング結果を保存する用のBitmapを用意
    result_bitmap.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());
    accum_bitmap.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());

    // LaunchParamsの設定
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.samples_per_launch = 1;
    params.max_depth = 8;

    initResultBufferOnDevice();

    // Camera settings
    camera.setOrigin(-0.327f, 0.777f, 2.22f);
    camera.setLookat(0.22f, 0.0f, 0.0f);
    camera.setUp(0.0f, 1.0f, 0.0f);
    camera.setFarClip(5000);
    camera.setFov(40.0f);
    camera.setAperture(0.04f);
    camera.setAspect((float)params.width / params.height);
    camera.setFocusDistance(2.5f);
    camera.enableTracking(pgGetCurrentWindow());

    // Raygen program
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, module, "__raygen__spectrum");
    // Shader binding table for raygen
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera = camera.getData();
    sbt.setRaygenRecord(raygen_record);

    // Lambda function to bind data and callables function to SBT
    auto setupCallable = [&](const Module& module, const std::string& dc, const std::string& cc)
    {
        EmptyRecord callable_record = {};
        auto [prg, id] = pipeline.createCallablesProgram(context, module, dc, cc);
        prg.recordPackHeader(&callable_record);
        sbt.addCallablesRecord(callable_record);
        return id;
    };

    // Callables programs for texture
    uint32_t constant_prg_id = setupCallable(module, DC_FUNC_TEXT("constant"), "");
    uint32_t bitmap_prg_id = setupCallable(module, DC_FUNC_TEXT("bitmap"), "");

    // Callables programs for surfaces
    // Diffuse
    uint32_t diffuse_prg_id = setupCallable(module, DC_FUNC_TEXT("sample_diffuse"), "");
    // Dielectric
    uint32_t dielectric_prg_id = setupCallable(module, DC_FUNC_TEXT("sample_dielectric"), "");
    // AreaEmitter
    uint32_t area_emitter_prg_id = setupCallable(module, DC_FUNC_TEXT("area_emitter"), "");

    SurfaceCallableID diffuse_id = { 0, diffuse_prg_id, 0 };
    SurfaceCallableID dielectric_id = { 0, dielectric_prg_id, 0 };
    SurfaceCallableID area_emitter_id = { 0, area_emitter_prg_id, 0 };

    // Shape用のCallableプログラム(主に面光源サンプリング用)
    uint32_t plane_sample_pdf_prg_id = setupCallable(module, DC_FUNC_TEXT("rnd_sample_plane"), CC_FUNC_TEXT("pdf_plane"));

    // 環境マッピング (Sphere mapping) 用のテクスチャとデータ準備
    auto env_texture = make_shared<FloatBitmapTexture>("resources/image/sepulchral_chapel_rotunda_4k.exr", bitmap_prg_id);
    env_texture->copyToDevice();
    env = EnvironmentEmitter{env_texture};
    env.copyToDevice();

    // Missプログラム
    ProgramGroup miss_prg = pipeline.createMissProgram(context, module, MS_FUNC_TEXT("envmap"));
    // Missプログラム用のShader Binding Tableデータ
    MissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env.devicePtr();
    sbt.setMissRecord({ miss_record });

    // Hitgroupプログラム
    // Plane
    auto plane_prg = pipeline.createHitgroupProgram(context, module, CH_FUNC_TEXT("plane"), IS_FUNC_TEXT("plane"));
    // Sphere
    auto sphere_prg = pipeline.createHitgroupProgram(context, module, CH_FUNC_TEXT("sphere"), IS_FUNC_TEXT("sphere"));
    // Triangle mesh
    auto mesh_prg = pipeline.createHitgroupProgram(context, module, CH_FUNC_TEXT("mesh"));

    struct Primitive
    {
        shared_ptr<Shape> shape;
        shared_ptr<Material> material;
        uint32_t sample_id;
    };

    uint32_t sbt_idx = 0;
    uint32_t sbt_offset = 0;
    uint32_t instance_id = 0;
    // ShapeとMaterialのデータをGPU上に準備しHitgroup用のSBTデータを追加するLambda関数
    auto setupPrimitive = [&](ProgramGroup& prg, const Primitive& primitive, const Matrix4f& transform)
        -> void
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
                .callable_id = primitive.material->surfaceCallableID(),
                .type = primitive.material->surfaceType()
            }
        };

        sbt.addHitgroupRecord({ record });
        sbt_idx++;

        // GASをビルドし、IASに追加
        ShapeInstance instance{primitive.shape->type(), primitive.shape, transform};
        instance.allowCompaction();
        instance.buildAccel(context, stream);
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);

        scene_ias.addInstance(instance);

        instance_id++;
        sbt_offset += SBT::NRay;
    };

    std::vector<AreaEmitterInfo> area_emitter_infos;
    // 面光源用のSBTデータを用意しグローバル情報としての光源情報を追加するLambda関数
    // 光源サンプリング時にCallable関数ではOptixInstanceに紐づいた行列情報を取得できないので
    // 行列情報をAreaEmitterInfoに一緒に設定しておく
    // ついでにShapeInstanceによって光源用のGASも追加
    auto setupAreaEmitter = [&](
        ProgramGroup& prg,
        shared_ptr<Shape> shape,
        AreaEmitter& area, const Matrix4f& transform, 
        uint32_t sample_pdf_id) 
        -> void
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
                .callable_id = area.surfaceCallableID(),
                .type = SurfaceType::AreaEmitter
            }
        };

        sbt_idx++;

        sbt.addHitgroupRecord({ record });

        // GASをビルドし、IASに追加
        ShapeInstance instance{shape->type(), shape, transform};
        instance.allowCompaction();
        instance.buildAccel(context, stream);
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);

        scene_ias.addInstance(instance);

        instance_id++;
        sbt_offset += SBT::NRay;

        AreaEmitterInfo area_emitter_info = 
        {
            .shape_data = shape->devicePtr(), 
            .objToWorld = transform,
            .worldToObj = transform.inverse(), 
            .sample_id = sample_pdf_id, 
            .gas_handle = instance.handle()
        };
        area_emitter_infos.push_back(area_emitter_info);
    };

    // Scene ---------------------------------------------------------
    constexpr int32_t N_SPD = 21;
    struct SPD {
        float lambda;
        float val;
    };

    array<SPD, N_SPD> red = 
    { { 
        {380.0f, 0.04f}, {397.0f, 0.04f}, {414.0f, 0.05f}, {431.0f, 0.05f}, {448.0f, 0.06f}, 
        {465.0f, 0.06f}, {482.0f, 0.06f}, {499.0f, 0.06f}, {516.0f, 0.06f}, {533.0f, 0.06f}, 
        {550.0f, 0.06f}, {567.0f, 0.07f}, {584.0f, 0.12f}, {601.0f, 0.29f}, {618.0f, 0.46f}, 
        {635.0f, 0.61f}, {652.0f, 0.61f}, {669.0f, 0.62f}, {686.0f, 0.15f}, {703.0f, 0.16f}, 
        {720.0f, 0.16f}
    } };

    array<SPD, N_SPD> green = 
    { { 
        {380.0f, 0.09f}, {397.0f, 0.09f}, {414.0f, 0.10f}, {431.0f, 0.10f}, {448.0f, 0.10f}, 
        {465.0f, 0.11f}, {482.0f, 0.13f}, {499.0f, 0.29f}, {516.0f, 0.46f}, {533.0f, 0.46f}, 
        {550.0f, 0.38f}, {567.0f, 0.30f}, {584.0f, 0.23f}, {601.0f, 0.17f}, {618.0f, 0.13f}, 
        {635.0f, 0.12f}, {652.0f, 0.12f}, {669.0f, 0.13f}, {686.0f, 0.15f}, {703.0f, 0.16f}, 
        {720.0f, 0.16f}
    } };

    auto createConstantTextureFromSPD = [&](const array<SPD, N_SPD>& spd)
        -> shared_ptr<ConstantTexture>
    {
        float lambda[N_SPD];
        float val[N_SPD];
        for (int32_t i = 0; i < N_SPD; i++)
        {
            lambda[i] = spd[i].lambda;
            val[i] = spd[i].val;
        }
        SampledSpectrum ret = SampledSpectrum::fromSample(lambda, val, N_SPD);
        return make_shared<ConstantTexture>(ret, constant_prg_id);
    };

    auto tex_white = make_shared<ConstantTexture>(SampledSpectrum::constant(0.95f), constant_prg_id);
    auto tex_red = createConstantTextureFromSPD(red);
    auto tex_green = createConstantTextureFromSPD(green);
    auto yellow = rgb2spectrum(Vec3f(0.7f, 0.7f, 0.3f));
    auto tex_yellow = make_shared<ConstantTexture>(yellow, constant_prg_id);
    auto tex_grid = make_shared<BitmapTexture>("resources/image/grid.png", bitmap_prg_id);

    unsigned int seed = tea<4>(0, 0);
    auto diamond = make_shared<TriangleMesh>("resources/model/diamond.obj");
    for (int i = 0; i < 15; i++)
    {
        const float scale = rnd(seed, 0.2f, 0.3f);
        const Vec3f axis = normalize(Vec3f(rnd(seed, -0.4, 0.4f), rnd(seed, -1.0f, 1.0f), rnd(seed, -0.4f, 0.4f)));
        const Vec3f pos = Vec3f(rnd(seed, -2.0f, 2.0f), scale * 0.9f, rnd(seed, -2.0f, 2.0f));
        const float rad = rnd(seed, 0.0f, math::pi);
        auto transform = Matrix4f::translate(pos) * Matrix4f::rotate(rad, axis) * Matrix4f::scale(scale);
        auto mat = make_shared<Dielectric>(dielectric_id, tex_white, 2.41f);
        mat->setSellmeierType(Sellmeier::Diamond);
        Primitive p{ diamond, mat, dielectric_prg_id };
        setupPrimitive(mesh_prg, p, transform);
    }

    // Floor
    {
        auto plane = make_shared<TriangleMesh>("resources/model/plane.obj");
        auto diffuse = make_shared<Diffuse>(diffuse_id, tex_grid);
        auto transform = Matrix4f::scale(3.0f);
        Primitive floor{ plane, diffuse };
        setupPrimitive(mesh_prg, floor, transform);
    }

    // Bunny
    {
        auto mesh = make_shared<TriangleMesh>("resources/model/bunny.obj");
        mesh->calculateNormalSmooth();
        auto diffuse = make_shared<Diffuse>(diffuse_id, tex_red);
        auto transform = Matrix4f::translate(-0.5, -0.07f, 0.0f) * Matrix4f::scale(2.0f);
        Primitive p{ mesh, diffuse, diffuse_prg_id };
        setupPrimitive(mesh_prg, p, transform);
    }

    // Armadillo
    {
        auto mesh = make_shared<TriangleMesh>("resources/model/Armadillo.ply");
        mesh->calculateNormalSmooth();
        auto diffuse = make_shared<Diffuse>(diffuse_id, tex_green);
        auto transform = Matrix4f::translate(-0.2f, 0.15f, -0.5f) * Matrix4f::rotate(math::pi * 7/6, {0,1,0}) * Matrix4f::scale(0.003f);
        Primitive p{ mesh, diffuse, diffuse_prg_id };
        setupPrimitive(mesh_prg, p, transform);
    }

    // Dragon
    {
        auto mesh = make_shared<TriangleMesh>("resources/model/dragon.obj");
        mesh->calculateNormalSmooth();
        auto diffuse = make_shared<Diffuse>(diffuse_id, tex_yellow);
        auto transform = Matrix4f::translate(0.5f, 0.11f, 0.7f) * Matrix4f::rotate(math::pi/3, {0,1,0}) * Matrix4f::scale(0.4f);
        Primitive p{ mesh, diffuse, diffuse_prg_id };
        setupPrimitive(mesh_prg, p, transform);
    }

    // Ceiling light
    {
        // Shape
        auto plane_light = make_shared<Plane>(Vec2f(-2.0f, -0.2f), Vec2f(2.0f, 0.2f));
        // Area emitter
        auto plane_area_emitter = AreaEmitter(area_emitter_id, tex_yellow, 20.0f);
        Matrix4f transform = Matrix4f::translate(0.0f, 3.0f, 0.0f);
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

    params.frame++;
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
        1);

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

    ImGui::Begin("Spectrum");

    ImGui::Text("Camera info:");
    ImGui::Text("Origin: %f %f %f", camera.origin().x(), camera.origin().y(), camera.origin().z());
    ImGui::Text("Lookat: %f %f %f", camera.lookat().x(), camera.lookat().y(), camera.lookat().z());
    ImGui::Text("Up: %f %f %f", camera.up().x(), camera.up().y(), camera.up().z());

    float farclip = camera.farClip();
    ImGui::SliderFloat("far clip", &farclip, 500.0f, 10000.0f);
    if (farclip != camera.farClip()) {
        camera.setFarClip(farclip);
        camera_update = true;
    }

    float aperture = camera.aperture();
    ImGui::SliderFloat("Aperture", &aperture, 0.01f, 4.0f);
    if (aperture != camera.aperture()) {
        camera.setAperture(aperture);
        camera_update = true;
    }

    float focus_dist = camera.focusDistance();
    ImGui::SliderFloat("Focus distance", &focus_dist, 1.0f, 100.0f);
    if (focus_dist != camera.focusDistance()) {
        camera.setFocusDistance(focus_dist);
        camera_update = true;
    }

    ImGui::Text("Frame rate: %.3f ms/frame (%.2f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Subframe index: %d", params.frame);

    ImGui::End();

    ImGui::Render();

    result_bitmap.draw(0, 0);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (params.frame == 20000) {
        result_bitmap.write(pgPathJoin(pgAppDir(), "spectrum.jpg"));
        pgExit();
    }
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
void App::keyPressed(int key)
{
    if (key == Key::Q)
    {
        pgExit();
    }
}

// ----------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    if (button != MouseButton::Middle) return;
    camera_update = true;
}

// ----------------------------------------------------------------
void App::mouseScrolled(float xoffset, float yoffset)
{
    camera_update = true;
}