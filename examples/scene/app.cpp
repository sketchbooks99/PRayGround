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
    params.max_depth = 10;
    params.frame = 0;
    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.devicePtr());

    CUDA_SYNC_CHECK();

    // Camera settings
    std::shared_ptr<Camera> camera(new Camera);
    camera->setOrigin(0, 300, 500);
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

    Callable area_emitter_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__area_emitter", "");
    scene.bindCallablesProgram(area_emitter_prg.program);

    SurfaceCallableID diffuse_id{ diffuse_sample_bsdf_prg.ID, diffuse_sample_bsdf_prg.ID, diffuse_pdf_prg.ID };
    SurfaceCallableID area_emitter_id{ area_emitter_prg.ID, area_emitter_prg.ID, area_emitter_prg.ID };

    // Miss program
    std::array<ProgramGroup, NRay> miss_prgs;
    miss_prgs[0] = pipeline.createMissProgram(context, module, "__miss__envmap");
    miss_prgs[1] = pipeline.createMissProgram(context, module, "__miss__shadow");
    scene.bindMissPrograms(miss_prgs);
    //scene.setEnvmap(std::make_shared<FloatBitmapTexture>("resources/image/drackenstein_quarry_4k.exr", bitmap_prg.ID));
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
    auto floor_checker = make_shared<CheckerTexture>(Vec3f(0.3f), Vec3f(0.8f), 10, checker_prg.ID);
    auto blue_constant = make_shared<ConstantTexture>(Vec3f(0.05f, 0.05f, 0.8f), constant_prg.ID);
    auto black_constant = make_shared<ConstantTexture>(Vec3f(0.05f), constant_prg.ID);

    // Materials
    auto green_diffuse = make_shared<Diffuse>(diffuse_id, green_constant);
    auto red_diffuse = make_shared<Diffuse>(diffuse_id, red_constant);
    auto white_diffuse = make_shared<Diffuse>(diffuse_id, white_constant);
    auto floor_diffuse = make_shared<Diffuse>(diffuse_id, floor_checker);
    auto blue_diffuse = make_shared<Diffuse>(diffuse_id, blue_constant);
    auto black_diffuse = make_shared<Diffuse>(diffuse_id, black_constant);

    // Shapes
    auto wall_plane = make_shared<Plane>(Vec2f(-25.0f), Vec2f(25.0f));
    
    // Groud floor
    scene.addObject("floor", make_shared<Plane>(Vec2f(-500, -500), Vec2f(500, 500)), 
        floor_diffuse, plane_prgs, Matrix4f::identity());

    // Sphere
    scene.addObject("sphere", make_shared<Sphere>(30),
        black_diffuse, sphere_prgs, Matrix4f::translate(0, 100, 0));

    // Carpet far with curves
    constexpr int32_t NUM_SEGMENTS = 10;
    constexpr float CURVE_HEIGHT = 100.0f;
    shared_ptr<Curves> floor_curves(new Curves(Curves::Type::CubicBSpline));

    uint32_t seed = tea<4>(0, 0);

    for (int x = -500; x <= 500; x += 25)
    {
        for (int z = -500; z <= 500; z += 25)
        {
            int32_t num_vertices_per_curve = NUM_SEGMENTS + (int32_t)Curves::getNumVertexPerSegment(floor_curves->curveType());
            float y_interval = CURVE_HEIGHT / (float)(num_vertices_per_curve - 1);
            int32_t base_idx = (int32_t)floor_curves->vertices().size();
            for (int s = 0; s < num_vertices_per_curve; s++)
            {
                float y = s * y_interval;
                const float coeff = (float)s / NUM_SEGMENTS;
                const float inv_coeff = (float)(num_vertices_per_curve - s) / num_vertices_per_curve;
                float xoffset = rnd(seed, -5, 5) * sinf(math::pi * coeff);
                float zoffset = rnd(seed, -5, 5) * cosf(math::pi * coeff);
                xoffset *= coeff;
                zoffset *= coeff;
                const Vec3f v(x + xoffset, y, z + zoffset);
                const float w = inv_coeff * 4.0f;
                
                floor_curves->addVertex(v);
                floor_curves->addWidth(w);

                if (s < (num_vertices_per_curve - (int32_t)Curves::getNumVertexPerSegment(floor_curves->curveType())))
                    floor_curves->addIndex(base_idx + s);
            }
        }
    }

    scene.addObject("carpet", floor_curves, red_diffuse, curve_prgs, Matrix4f::identity());

    //scene.addObject("box", make_shared<Box>(Vec3f(-15), Vec3f(15)),
    //    green_diffuse, box_prgs, Matrix4f::translate(50, 50, 0));

    scene.addLight("ceiling_light", make_shared<Plane>(Vec2f(-100.0f), Vec2f(100.0f)),
        make_shared<AreaEmitter>(area_emitter_id, make_shared<ConstantTexture>(Vec3f(1.0f), constant_prg.ID), 5.0f), 
        plane_prgs, Matrix4f::translate(0, 300.0f, 0));

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

}

// ------------------------------------------------------------------
void App::keyReleased(int key)
{

}



