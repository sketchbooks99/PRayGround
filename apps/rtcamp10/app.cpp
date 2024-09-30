#include "app.h"

#include <chrono>

#define SUBMISSION 0

void App::initResultBufferOnDevice() 
{
    params.frame = 0u;

    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();

    params.result_buffer = (Vec4u*)result_bmp.deviceData();
    params.accum_buffer = (Vec4f*)accum_bmp.deviceData();
}

void App::handleCameraUpdate() 
{
    if (!is_camera_updated) return;
    is_camera_updated = false;

    scene.updateSBT(+(SBTRecordType::Raygen));

    initResultBufferOnDevice();
}

// ------------------------------------------------------------------
void App::setup()
{
    using namespace std::chrono;

    // Initialize CUDA 
    stream = 0;
    CUDA_CHECK(cudaFree(0));

    // Initialize OptiX
    OPTIX_CHECK(optixInit());
    ctx.disableValidation();
    ctx.create();

    // Initialize pipeline
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(10);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);

    // Create module
    Module module = pipeline.createModuleFromPtxFile(ctx, "ptx/rtcamp10_generated_kernels.cu.ptx");

    // Initialize bitmap 
    const int32_t width = pgGetWidth();
    const int32_t height = pgGetHeight();
    result_bmp.allocate(PixelFormat::RGBA, width, height);
    accum_bmp.allocate(PixelFormat::RGBA, width, height);
    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();

    // Launch parameter initialization
    params.width = width;
    params.height = height;
    params.samples_per_launch = 8;
    params.frame = 0u;
    params.max_depth = 10;
    params.result_buffer = (Vec4u*)result_bmp.deviceData();
    params.accum_buffer = (Vec4f*)accum_bmp.deviceData();

    // Setup scene
    Scene<Camera, NRay>::AccelSettings accel_settings;
    accel_settings.allow_accel_compaction = true;
    accel_settings.allow_accel_update = true;
    scene.setup(accel_settings);

    // Setup camera
    shared_ptr<Camera> camera = make_shared<Camera>();
    camera->setOrigin(0, -50, 50);
    camera->setLookat(0, -80, 0);
    camera->setUp(0, 1, 0);
    camera->setFov(40);
    camera->setAspect((float)width / height);
    camera->enableTracking(pgGetCurrentWindow());
    scene.setCamera(camera);

    // Raygen program
    ProgramGroup raygen = pipeline.createRaygenProgram(ctx, module, "__raygen__pinhole");
    scene.bindRaygenProgram(raygen);

    struct Callable {
        Callable(const pair<ProgramGroup, uint32_t>& callable)
            : program(callable.first), ID(callable.second) {};
        ProgramGroup program;
        uint32_t ID;
    };

    auto setupCallable = [&](const string& dc, const string& cc) -> uint32_t {
        Callable callable = pipeline.createCallablesProgram(ctx, module, dc, cc);
        scene.bindCallablesProgram(callable.program);
        return callable.ID;
    };

    // Texture programs
    auto bitmap_id = setupCallable("__direct_callable__bitmap", "");
    auto checker_id = setupCallable("__direct_callable__checker", "");
    auto constant_id = setupCallable("__direct_callable__constant", "");

    // Miss program
    ProgramGroup miss = pipeline.createMissProgram(ctx, module, "__miss__envmap");
    ProgramGroup miss_shadow = pipeline.createMissProgram(ctx, module, "__miss__shadow");
    scene.bindMissPrograms({ miss, miss_shadow });
    auto envmap_texture = make_shared<FloatBitmapTexture>("resources/image/drackenstein_quarry_4k.exr", bitmap_id);
    scene.setEnvmap(envmap_texture);
    // TODO: Create Distribution2D for envmap importance sampling

    // Hitgroup programs
    // Mesh
    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__mesh");
    ProgramGroup mesh_shadow_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__shadow");
    // Sphere
    ProgramGroup sphere_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__custom", "__intersection__sphere");
    ProgramGroup sphere_shadow_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__shadow", "__intersection__sphere");
    // Plane
    ProgramGroup plane_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__custom", "__intersection__plane");
    ProgramGroup plane_shadow_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__shadow", "__intersection__plane");

    // Surface programs
    SurfaceCallableID diffuse_id = { 
        setupCallable("__direct_callable__sample_diffuse", ""), 
        setupCallable("__direct_callable__bsdf_diffuse", ""),
        setupCallable("__direct_callable__pdf_diffuse", "")
    };
    SurfaceCallableID conductor_id = {
        setupCallable("__direct_callable__sample_conductor", ""),
        setupCallable("__direct_callable__bsdf_conductor", ""),
        setupCallable("__direct_callable__pdf_conductor", "")
    };
    SurfaceCallableID dielectric_id = {
        setupCallable("__direct_callable__sample_dielectric", ""),
        setupCallable("__direct_callable__bsdf_dielectric", ""),
        setupCallable("__direct_callable__pdf_dielectric", "")
    };
    SurfaceCallableID disney_id = {
        setupCallable("__direct_callable__sample_disney", ""),
        setupCallable("__direct_callable__bsdf_disney", ""),
        setupCallable("__direct_callable__pdf_disney", "")
    };
    SurfaceCallableID layered_id = {
        setupCallable("__direct_callable__sample_layered", ""),
        setupCallable("__direct_callable__bsdf_layered", ""),
        setupCallable("__direct_callable__pdf_layered", "")
    };
    SurfaceCallableID area_emitter_id = { 
        setupCallable("__direct_callable__area_emitter", "")
    };
    PG_LOG("diffuse_id:", diffuse_id);
    PG_LOG("conductor_id:", conductor_id);
    PG_LOG("dielectric_id:", dielectric_id);
    PG_LOG("disney_id:", disney_id);
    PG_LOG("layered_id:", layered_id);
    PG_LOG("area_emitter_id:", area_emitter_id);

    // Create surfaces
    auto floor_mat = make_shared<Diffuse>(diffuse_id, make_shared<CheckerTexture>(Vec3f(0.8f), Vec3f(0.2f), 10, checker_id));
    auto sphere_mat = make_shared<Dielectric>(dielectric_id,
        /* texture = */ make_shared<ConstantTexture>(Vec3f(1.0f, 1.0f, 1.0f), constant_id),
        /* ior = */ 1.33f,
        /* absorb_coeff = */ 0.0f,
        /* sellmeier = */ Sellmeier::None, 
        Thinfilm(
            /* ior = */ Vec3f(1.33f),
            /* thickness = */ make_shared<ConstantTexture>(Vec3f(1.0f), constant_id),
            /* thickness_scale = */ 550.0f,
            /* tf_ior = */ 1.7f,
            /* extinction = */ Vec3f(1.0f)
        ));
    auto bunny_mat = make_shared<Conductor>(conductor_id, 
        /* texture = */ make_shared<ConstantTexture>(Vec3f(0.8f, 0.8f, 0.3f), constant_id), 
        /* twosided = */ true, 
        Thinfilm(
            /* ior = */ Vec3f(1.9f),
            /* thickness = */ make_shared<ConstantTexture>(Vec3f(1.0f), constant_id),
            /* thickness_scale = */ 550.0f,
            /* tf_ior = */ 1.33f,
            /* extinction = */ Vec3f(1.5f)
        ));

    auto katsuo_base = make_shared<Disney>(disney_id, make_shared<BitmapTexture>("Katsuo Color.png", bitmap_id));
    katsuo_base->setSubsurface(0.8f);
    katsuo_base->setMetallic(0.8f);
    katsuo_base->setRoughness(0.2f);
    auto katsuo_oil = make_shared<Dielectric>(dielectric_id,
        make_shared<ConstantTexture>(Vec3f(0.9f, 0.9f, 0.8f), constant_id),
        1.5f, 0.0f, Sellmeier::None,
        Thinfilm(Vec3f(1.5f), make_shared<BitmapTexture>("Katsuo thinfilm thickness.png", bitmap_id), 200.0f, 1.33f, Vec3f(3.0f, 5.0f, 6.0f)));
    std::vector <std::shared_ptr<Material>> layers = { katsuo_oil, katsuo_base };
    auto katsuo_mat = make_shared<Layered>(layered_id, layers);
    katsuo_mat->setBumpmap(make_shared<BitmapTexture>("Katsuo Normal.png", bitmap_id));

    // Katsuo 2
    auto katsuo2_mat = make_shared<Disney>(disney_id, make_shared<BitmapTexture>("Katsuo Color.png", bitmap_id));
    katsuo2_mat->setSubsurface(0.8f);
    katsuo2_mat->setMetallic(0.8f);
    katsuo2_mat->setRoughness(0.2f);
    katsuo2_mat->setBumpmap(make_shared<BitmapTexture>("Katsuo Normal.png", bitmap_id));

    // Setup geometries in the scene
    // Floor geometry
    shared_ptr<Plane> plane = make_shared<Plane>(Vec2f(-200), Vec2f(200));
    scene.addObject("floor", plane, floor_mat, { plane_prg, plane_shadow_prg }, Matrix4f::translate(0, -100, 0));

    // Sphere geometry
    shared_ptr<Sphere> sphere = make_shared<Sphere>(10);
    scene.addObject("sphere", sphere, sphere_mat, { sphere_prg, sphere_shadow_prg }, Matrix4f::translate(-10, -80, 0));

    shared_ptr<TriangleMesh> bunny = make_shared<TriangleMesh>("uv_bunny.obj");
    bunny->calculateNormalSmooth();
    scene.addObject("bunny", bunny, bunny_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(10, -80, 0) * Matrix4f::scale(100));

    shared_ptr<TriangleMesh> katsuo = make_shared<TriangleMesh>("Katsuo.obj");
    scene.addObject("katsuo", katsuo, katsuo_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(30, -80, 0) * Matrix4f::scale(10));

    shared_ptr<TriangleMesh> katsuo2 = make_shared<TriangleMesh>("Katsuo.obj");
    scene.addObject("katsuo", katsuo2, katsuo2_mat, { mesh_prg, mesh_shadow_prg }, Matrix4f::translate(60, -80, 0)* Matrix4f::scale(10));
    


    // Setup context
    CUDA_CHECK(cudaStreamCreate(&stream));
    scene.copyDataToDevice();
    scene.buildAccel(ctx, stream);
    scene.buildSBT();
    pipeline.create(ctx);

    params.handle = scene.accelHandle();

    // GUI initialization
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    const char* glsl_version = "#version 330";
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(pgGetCurrentWindow()->windowPtr(), true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}


// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();

    scene.launchRay(ctx, pipeline, params, stream, result_bmp.width(), result_bmp.height(), 1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    result_bmp.copyFromDevice();

    params.frame++;
}

// ------------------------------------------------------------------
void App::draw()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("RTCAMP 10 editor");

    // Update katsuo object
    auto katsuo = scene.getObject("katsuo");
    auto katsuo_mat = dynamic_pointer_cast<Layered>(katsuo->materials[0]);
    auto katsuo_top_layer = dynamic_pointer_cast<Dielectric>(katsuo_mat->layerAt(0));
    Thinfilm thinfilm = katsuo_top_layer->thinfilm();
    Vec3f ior = thinfilm.ior();
    Vec3f extinction = thinfilm.extinction();
    float thickness_scale = thinfilm.thicknessScale();
    float tf_ior = thinfilm.tfIor();

    if (ImGui::TreeNode("Katsuo")) {
        if (ImGui::SliderFloat3("IOR", &ior[0], 1.0f, 20.0f)) {
            thinfilm.setIor(ior);
            katsuo_top_layer->setThinfilm(thinfilm);
            is_camera_updated = true;
        }
        if (ImGui::SliderFloat3("Extinction", &extinction[0], 0.0f, 20.0f)) {
            thinfilm.setExtinction(extinction);
            katsuo_top_layer->setThinfilm(thinfilm);
            is_camera_updated = true;
        }
        if (ImGui::SliderFloat("Thickness Scale", &thickness_scale, 0.0f, 1000.0f)) {
            thinfilm.setThicknessScale(thickness_scale);
            katsuo_top_layer->setThinfilm(thinfilm);
            is_camera_updated = true;
        }
        if (ImGui::SliderFloat("TF IOR", &tf_ior, 1.0f, 20.0f)) {
            thinfilm.setTfIor(tf_ior);
            katsuo_top_layer->setThinfilm(thinfilm);
            is_camera_updated = true;
        }

        ImGui::TreePop();
    }
    katsuo_top_layer->copyToDevice();


    // Update bunny 
    //auto bunny = scene.getObject("bunny");
    //auto bunny_mat = dynamic_pointer_cast<Conductor>(bunny->materials[0]);
    //
    //Vec3f ior = bunny_mat->ior();
    //auto tf_thickness_ptr = dynamic_pointer_cast<ConstantTexture>(bunny_mat->thinfilmThickness());
    //Vec3f tf_thickness = tf_thickness_ptr->color();
    //float tf_ior = bunny_mat->thinfilmIOR();
    //Vec3f extinction = bunny_mat->extinction();
    //if (ImGui::TreeNode("Bunny")) {
    //    if (ImGui::SliderFloat3("IOR", &ior[0], 1.0f, 20.0f)) {
    //        bunny_mat->setIOR(ior);
    //        is_camera_updated = true;
    //    }
    //    if (ImGui::SliderFloat("TF IOR", &tf_ior, 1.0f, 20.0f)) {
    //        bunny_mat->setThinfilmIOR(tf_ior);
    //        is_camera_updated = true;
    //    }
    //    if (ImGui::SliderFloat("TF Thickness", &tf_thickness.x(), 0.0f, 1000.0f)) {
    //        tf_thickness_ptr->setColor(tf_thickness);
    //        bunny_mat->setThinfilmThickness(tf_thickness_ptr);
    //        is_camera_updated = true;
    //    }
    //    if (ImGui::SliderFloat3("Extinction", &extinction[0], 0.0f, 20.0f)) {
    //        bunny_mat->setExtinction(extinction);
    //        is_camera_updated = true;
    //    }

    //    ImGui::TreePop();
    //}
    //bunny_mat->copyToDevice();

    //// Update sphere parameters
    //auto sphere = scene.getObject("sphere");
    //auto sphere_mat = dynamic_pointer_cast<Dielectric>(sphere->materials[0]);

    //float ior_sphere = sphere_mat->ior();
    //auto tf_thickness_sphere_ptr = dynamic_pointer_cast<ConstantTexture>(sphere_mat->thinfilmThickness());
    //Vec3f tf_thickness_sphere = tf_thickness_sphere_ptr->color();
    //float tf_ior_sphere = sphere_mat->thinfilmIOR();
    //Vec3f extinction_sphere = sphere_mat->extinction();
    //if (ImGui::TreeNode("Sphere")) {
    //    if (ImGui::SliderFloat("IOR", &ior_sphere, 1.0f, 20.0f)) {
    //        sphere_mat->setIor(ior_sphere);
    //        is_camera_updated = true;
    //    }
    //    if (ImGui::SliderFloat("TF IOR", &tf_ior_sphere, 1.0f, 20.0f)) {
    //        sphere_mat->setThinfilmIOR(tf_ior_sphere);
    //        is_camera_updated = true;
    //    }
    //    if (ImGui::SliderFloat("TF Thickness", &tf_thickness_sphere.x(), 0.0f, 1000.0f)) {
    //        tf_thickness_sphere_ptr->setColor(tf_thickness_sphere);
    //        sphere_mat->setThinfilmThickness(tf_thickness_sphere_ptr);
    //        is_camera_updated = true;
    //    }
    //    if (ImGui::SliderFloat3("Extinction", &extinction_sphere[0], 0.0f, 20.0f)) {
    //        sphere_mat->setExtinction(extinction_sphere);
    //        is_camera_updated = true;
    //    }
    //    ImGui::TreePop();
    //}
    //sphere_mat->copyToDevice();

        

    ImGui::End();
    ImGui::Render();

    result_bmp.draw();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// ------------------------------------------------------------------
void App::mousePressed(float x, float y, int button)
{
    
}

// ------------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    if (button == MouseButton::Middle) is_camera_updated = true;
}

// ------------------------------------------------------------------
void App::mouseReleased(float x, float y, int button)
{
    
}

// ------------------------------------------------------------------
void App::mouseMoved(float x, float y)
{
    
}

// ------------------------------------------------------------------
void App::mouseScrolled(float x, float y)
{
    is_camera_updated = true;
}

// ------------------------------------------------------------------
void App::keyPressed(int key)
{

}

// ------------------------------------------------------------------
void App::keyReleased(int key)
{

}



