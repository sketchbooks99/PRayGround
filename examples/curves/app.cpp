#include "app.h"

void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();
    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.devicePtr());

    CUDA_SYNC_CHECK();
}

void App::handleCameraUpdate()
{
    if (!is_camera_updated)
        return;
    is_camera_updated = false;

    scene.updateSBT(+(SBTRecordType::Raygen));

    initResultBufferOnDevice();
}

// ------------------------------------------------------------------
void App::setup()
{
    // Initialize CUDA
    stream = 0; 
    CUDA_CHECK(cudaFree(0));

    // Initialize OptixDeviceContext
    OPTIX_CHECK(optixInit());
    context.disableValidation();
    context.create();

    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(5);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);

    // Create module
    Module module;
    module = pipeline.createModuleFromCudaFile(context, "kernels.cu");

    const int32_t width = pgGetWidth();
    const int32_t height = pgGetHeight();
    result_bmp.allocate(PixelFormat::RGBA, width, height);
    result_bmp.allocateDevicePtr();
    accum_bmp.allocate(PixelFormat::RGBA, width, height);
    accum_bmp.allocateDevicePtr();

    // Configuration of launch parameters
    params.width = result_bmp.width();
    params.height = result_bmp.height();
    params.samples_per_launch = 1;
    params.frame = 0;
    params.max_depth = 5;
    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.devicePtr());

    // Camera settings
    std::shared_ptr<Camera> camera(new Camera);
    camera->setOrigin(175, 192, 405);
    camera->setLookat(0, 0, 0);
    camera->setUp(0, 1, 0);
    camera->setFov(40);
    camera->setAspect((float)width / height);
    camera->enableTracking(pgGetCurrentWindow());

    // Raygen program
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, module, "__raygen__pinhole");
    scene.bindRaygenProgram(raygen_prg);
    scene.setCamera(camera);

    // Create callables program for texture
    struct Callable
    {
        Callable(const std::pair<ProgramGroup, uint32_t>& callable)
            : program(callable.first), ID(callable.second) {}
        ProgramGroup program; 
        uint32_t ID;
    };
    Callable bitmap_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__bitmap", "");
    Callable constant_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__constant", "");
    Callable checker_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__checker", "");
    scene.bindCallablesProgram(bitmap_prg.program);
    scene.bindCallablesProgram(constant_prg.program);
    scene.bindCallablesProgram(checker_prg.program);

    // Create callables program for surfaces
    Callable diffuse_sample_bsdf_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__sample_diffuse", "__continuation_callable__bsdf_diffuse");
    Callable diffuse_pdf_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__pdf_diffuse", "");
    scene.bindCallablesProgram(diffuse_sample_bsdf_prg.program);
    scene.bindCallablesProgram(diffuse_pdf_prg.program);

    Callable glass_sample_bsdf_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__sample_glass", "__continuation_callable__bsdf_glass");
    Callable glass_pdf_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__pdf_glass", "");
    scene.bindCallablesProgram(glass_sample_bsdf_prg.program);
    scene.bindCallablesProgram(glass_pdf_prg.program);

    Callable disney_sample_bsdf_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__sample_disney", "__continuation_callable__bsdf_disney");
    Callable disney_pdf_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__pdf_disney", "");
    scene.bindCallablesProgram(disney_sample_bsdf_prg.program);
    scene.bindCallablesProgram(disney_pdf_prg.program);

    Callable area_emitter_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__area_emitter", "");
    scene.bindCallablesProgram(area_emitter_prg.program);

    SurfaceCallableID diffuse_id{ diffuse_sample_bsdf_prg.ID, diffuse_sample_bsdf_prg.ID, diffuse_pdf_prg.ID };
    SurfaceCallableID glass_id{ glass_sample_bsdf_prg.ID, glass_sample_bsdf_prg.ID, glass_pdf_prg.ID };
    SurfaceCallableID area_emitter_id{ area_emitter_prg.ID, area_emitter_prg.ID, area_emitter_prg.ID };
    SurfaceCallableID disney_id{ disney_sample_bsdf_prg.ID, disney_sample_bsdf_prg.ID, disney_pdf_prg.ID };

    // Miss program
    std::array<ProgramGroup, NRay> miss_prgs;
    miss_prgs[0] = pipeline.createMissProgram(context, module, "__miss__envmap");
    miss_prgs[1] = pipeline.createMissProgram(context, module, "__miss__shadow");
    scene.bindMissPrograms(miss_prgs);
    scene.setEnvmap(make_shared<ConstantTexture>(Vec3f(0.0f), constant_prg.ID));

    // Hitgroup program
    std::array<ProgramGroup, NRay> mesh_prgs;
    mesh_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh");
    mesh_prgs[1] = pipeline.createHitgroupProgram(context, module, "__closesthit__shadow");

    std::array<ProgramGroup, NRay> sphere_prgs;
    sphere_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", IS_FUNC_TEXT("pg_sphere"));
    sphere_prgs[1] = pipeline.createHitgroupProgram(context, module, "__closesthit__shadow", IS_FUNC_TEXT("pg_sphere"));

    std::array<ProgramGroup, NRay> plane_prgs;
    plane_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", IS_FUNC_TEXT("pg_plane"));
    plane_prgs[1] = pipeline.createHitgroupProgram(context, module, "__closesthit__shadow", IS_FUNC_TEXT("pg_plane"));

    std::array<ProgramGroup, NRay> box_prgs;
    box_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", IS_FUNC_TEXT("pg_box"));
    box_prgs[1] = pipeline.createHitgroupProgram(context, module, "__closesthit__shadow", IS_FUNC_TEXT("pg_box"));

    std::array<ProgramGroup, NRay> curve_prgs;
    Module curve_module = pipeline.createBuiltinIntersectionModule(context, OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE);
    curve_prgs[0] = pipeline.createHitgroupProgram(context, { module, "__closesthit__curves" }, { curve_module, "" });
    curve_prgs[1] = pipeline.createHitgroupProgram(context, { module, "__closesthit__shadow" }, { curve_module, "" });

    // Textures
    auto green_constant = make_shared<ConstantTexture>(Vec3f(0.05f, 0.8f, 0.05f), constant_prg.ID);
    auto red_constant = make_shared<ConstantTexture>(Vec3f(0.8f, 0.05f, 0.05f), constant_prg.ID);
    auto white_constant = make_shared<ConstantTexture>(Vec3f(0.8f), constant_prg.ID);
    auto floor_checker = make_shared<CheckerTexture>(Vec3f(0.8f), Vec3f(0.3f), 10, checker_prg.ID);
    auto blue_constant = make_shared<ConstantTexture>(Vec3f(0.05f, 0.05f, 0.8f), constant_prg.ID);
    auto yellow_constant = make_shared<ConstantTexture>(Vec3f(0.8f, 0.7f, 0.3f), constant_prg.ID);
    auto black_constant = make_shared<ConstantTexture>(Vec3f(0.01f), constant_prg.ID);

    // Materials
    auto green_diffuse = make_shared<Diffuse>(diffuse_id, green_constant);
    //auto red_diffuse = make_shared<Diffuse>(diffuse_id, red_constant);
    auto green_disney = make_shared<Disney>(disney_id, green_constant);
    green_disney->setMetallic(0.6f);
    green_disney->setRoughness(0.05f);
    green_disney->setSubsurface(0.1f);
    green_disney->setAnisotropic(0.8f);
    auto red_diffuse = make_shared<Diffuse>(diffuse_id, red_constant);
    auto yellow_diffuse = make_shared<Diffuse>(diffuse_id, yellow_constant);
    auto black_diffuse = make_shared<Diffuse>(diffuse_id, black_constant);
    auto white_glass = make_shared<Dielectric>(glass_id, white_constant, 1.5f);
    auto white_diffuse = make_shared<Diffuse>(diffuse_id, white_constant);
    auto checker_diffuse = make_shared<Diffuse>(diffuse_id, floor_checker);

    // Shapes
    auto wall_plane = make_shared<Plane>(Vec2f(-25.0f), Vec2f(25.0f));

    // Carpet far with curves
    constexpr int32_t NUM_SEGMENTS = 10;
    constexpr float CURVE_HEIGHT = 100.0f;
    shared_ptr<Curves> curves(new Curves(Curves::Type::CubicBSpline));

    uint32_t seed = tea<4>(0, 0);

    constexpr float CURVE_LENGTH = 1000.0f;
    PerlinNoise pnoise(seed);
    for (int z = -500; z <= 500; z += 5)
    {
        // The number of vertices per curve
        int32_t n_vertices = NUM_SEGMENTS + (int32_t)Curves::getNumVertexPerSegment(curves->curveType());
        float x_interval = CURVE_LENGTH / (float)(n_vertices - 1);
        int32_t base_idx = (int32_t)curves->vertices().size();
        for (int s = 0; s < n_vertices; s++)
        {
            float x = -500.0f + s * x_interval;
            Vec3f p(x, 0, float(z));
            const float offset = (float)s / NUM_SEGMENTS;

            auto getOffsetedNoise = [pnoise, p](uint32_t& seed) -> float
            {
                return pnoise.noise(p + UniformSampler::get3D(seed));
            };

            const float radius = lerp(10.0f, 30.0f, getOffsetedNoise(seed)) * offset;
            const float y = lerp(-100.f, 100.0f, getOffsetedNoise(seed));
            const float xoffset = lerp(-10.0f, 10.0f, getOffsetedNoise(seed));
            const float zoffset = lerp(-5.0f, 5.0f, getOffsetedNoise(seed));
            const Vec3f v(x + xoffset, y, z);
            
            curves->addVertex(v);
            curves->addWidth(radius);

            if (s < (n_vertices - (int32_t)Curves::getNumVertexPerSegment(curves->curveType())))
                curves->addIndex(base_idx + s);
        }
    }

    // Curve primitives
    scene.addObject("curves", curves, red_diffuse, curve_prgs, Matrix4f::identity());

    // Sphere
    scene.addObject("sphere", make_shared<Sphere>(50),
        black_diffuse, sphere_prgs, Matrix4f::translate(0, 50, 0));

    auto bunny = make_shared<TriangleMesh>("resources/model/bunny.obj");
    bunny->calculateNormalSmooth();
    scene.addObject("bunny", bunny,
        green_disney, mesh_prgs, Matrix4f::translate(-80, 50, 80) * Matrix4f::rotate(math::pi / 12.0f, Vec3f(0.5f, 0.0f, 0.5f)) * Matrix4f::scale(500));

    auto teapot = make_shared<TriangleMesh>("resources/model/teapot.obj");
    teapot->calculateNormalSmooth();
    scene.addObject("teapot", teapot,
        yellow_diffuse, mesh_prgs, Matrix4f::translate(100, 30, -80) * Matrix4f::rotate(math::pi / 12.0f, Vec3f(0.0f, 0.0f, 1.0f)) * Matrix4f::scale(20));

    scene.addObject("box", make_shared<Box>(Vec3f(-500, -200, -500), Vec3f(500, 500, 500)),
        checker_diffuse, box_prgs, Matrix4f::identity());

    // Light 
    scene.addLight("ceiling_light", make_shared<Plane>(Vec2f(-100.0f), Vec2f(100.0f)),
        make_shared<AreaEmitter>(area_emitter_id, make_shared<ConstantTexture>(Vec3f(1.0f), constant_prg.ID), 5.0f), 
        plane_prgs, Matrix4f::translate(0, 300, 0));

    CUDA_CHECK(cudaStreamCreate(&stream));
    scene.copyDataToDevice();
    scene.buildAccel(context, stream);
    scene.buildSBT();
    pipeline.create(context);

    params.handle = scene.accelHandle();
}

// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();

    scene.launchRay(context, pipeline, params, stream, result_bmp.width(), result_bmp.height(), 1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    params.frame++;

    result_bmp.copyFromDevice();

    std::string msg = "origin: " + toString(scene.camera()->origin()) + ", lookat: " + toString(scene.camera()->lookat()) + ", samples: " + toString(params.frame);
    pgSetWindowName(msg);

    if (params.frame == 5000)
        result_bmp.write(pgPathJoin(pgAppDir(), "curves.jpg"));
}

// ------------------------------------------------------------------
void App::draw()
{
    result_bmp.draw(0, 0);
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
    if (key == Key::S)
        result_bmp.write(pgPathJoin(pgAppDir(), "curves.jpg"));
}

// ------------------------------------------------------------------
void App::keyReleased(int key)
{

}



