#include "app.h"

// ----------------------------------------------------------------
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
    params.max_depth = 10;
    params.white = 1.0f;

    initResultBufferOnDevice();

    // カメラの設定
    camera.setOrigin(10.0f, 5.0f, 0.0f);
    camera.setLookat(0.0f, 2.0f, 0.0f);
    camera.setUp(0.0f, 1.0f, 0.0f);
    camera.setFarClip(5000);
    camera.setFov(40.0f);
    camera.setAspect(static_cast<float>(params.width) / params.height);
    camera.enableTracking(pgGetCurrentWindow());

    // Raygenプログラム
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, module, "__raygen__pinhole");
    // Raygenプログラム用のShader Binding Tableデータ
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera = camera.getData();
    sbt.setRaygenRecord(raygen_record);

    // Callable関数とShader Binding TableにCallable関数用のデータを登録するLambda関数
    auto setupCallable = [&](const std::string& dc, const std::string& cc)
    {
        EmptyRecord callable_record = {};
        auto [prg, id] = pipeline.createCallablesProgram(context, module, dc, cc);
        prg.recordPackHeader(&callable_record);
        sbt.addCallablesRecord(callable_record);
        return id;
    };

    // テクスチャ用のCallableプログラム
    uint32_t constant_prg_id = setupCallable(DC_FUNC_TEXT("constant"), "");
    uint32_t bitmap_prg_id = setupCallable(DC_FUNC_TEXT("bitmap"), "");

    // Surface用のCallableプログラム 
    // Diffuse
    uint32_t diffuse_sample_bsdf_prg_id = setupCallable(DC_FUNC_TEXT("sample_diffuse"), CC_FUNC_TEXT("bsdf_diffuse"));
    uint32_t diffuse_pdf_prg_id = setupCallable(DC_FUNC_TEXT("pdf_diffuse"), "");
    SurfaceCallableID diffuse_id = { diffuse_sample_bsdf_prg_id, diffuse_sample_bsdf_prg_id, diffuse_pdf_prg_id };

    // 環境マッピング (Sphere mapping) 用のテクスチャとデータ準備
    auto env_texture = make_shared<ConstantTexture>(Vec3f(3.0f), constant_prg_id);
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
    // Triangle mesh
    auto mesh_prg = pipeline.createHitgroupProgram(context, module, CH_FUNC_TEXT("mesh"));

    uint32_t sbt_idx = 0;

    // Objファイルからメッシュ読み込み
    shared_ptr<TriangleMesh> mesh(new TriangleMesh());
    mesh->loadWithMtl("resources/model/sponza/sponza.obj", material_attributes);
    mesh->copyToDevice();
    mesh->offsetSbtIndex(sbt_idx);

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.normalizedCoords = 1;
    tex_desc.sRGB = 1;

    // 読み込んだマテリアル情報からDiffuseマテリアルを生成し、Shader binding tableを構築
    for (const auto& ma : material_attributes)
    {
        shared_ptr<Texture> texture;
        // Diffuseテクスチャが読み込めている場合はBitmapTextureでテクスチャを初期化
        std::string diffuse_texname = ma.findOneString("diffuse_texture", "");
        if (!diffuse_texname.empty())
            texture = make_shared<BitmapTexture>(diffuse_texname, tex_desc, bitmap_prg_id);
        // テクスチャがない場合は単色テクスチャを生成
        else
            texture = make_shared<ConstantTexture>(ma.findOneVec3f("diffuse", Vec3f(0.0f)), constant_prg_id);
        texture->copyToDevice();
        auto diffuse = make_shared<Diffuse>(diffuse_id, texture);
        diffuse->copyToDevice();

        HitgroupRecord record;
        mesh_prg.recordPackHeader(&record);
        record.data =
        {
            .shape_data = mesh->devicePtr(),
            .surface_info = {
                .data = diffuse->devicePtr(),
                .callable_id = diffuse->surfaceCallableID(),
                .type = diffuse->surfaceType()
            }
        };

        sbt.addHitgroupRecord({ record });
    }

    // Geometry acceleration structureの構築　
    GeometryAccel gas{ ShapeType::Mesh };
    gas.addShape(mesh);
    gas.allowCompaction();
    gas.build(context, stream);

    CUDA_CHECK(cudaStreamCreate(&stream));
    //scene_ias.build(context, stream);
    sbt.createOnDevice();
    params.handle = gas.handle();
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

    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

    float start_time = pgGetElapsedTimef();

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
    params.frame++;

    render_time = pgGetElapsedTimef() - start_time;

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

    ImGui::Begin("Path tracing GUI");

    ImGui::SliderFloat("White", &params.white, 0.01f, 1.0f);
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

    ImGui::Text("Frame rate: %.3f ms/frame (%.2f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Render time: %.3f ms/frame", render_time * 1000.0f);
    ImGui::Text("Subframe index: %d", params.frame);

    ImGui::End();

    ImGui::Render();

    result_bitmap.draw(0, 0, pgGetWidth(), pgGetHeight());

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (params.frame == 4096)
        result_bitmap.write(pgPathJoin(pgAppDir(), "objscene.jpg"));
}

// ----------------------------------------------------------------
void App::close()
{
    env.free();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    pipeline.destroy();
    context.destroy();
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
