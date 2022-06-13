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
    camera->setOrigin(0, 0, 100);
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
    SurfaceCallableID glass_id{ glass_sample_bsdf_prg.ID, glass_sample_bsdf_prg.ID, glass_pdf_prg.ID };
    SurfaceCallableID area_emitter_id{ area_emitter_prg.ID, area_emitter_prg.ID, area_emitter_prg.ID };

    // Miss program
    std::array<ProgramGroup, NRay> miss_prgs;
    miss_prgs[0] = pipeline.createMissProgram(context, module, "__miss__envmap");
    miss_prgs[1] = pipeline.createMissProgram(context, module, "__miss__shadow");
    scene.bindMissPrograms(miss_prgs);
    scene.setEnvmap(std::make_shared<FloatBitmapTexture>("resources/image/drackenstein_quarry_4k.exr", bitmap_prg.ID));

    // Hitgroup program
    std::array<ProgramGroup, NRay> mesh_prgs;
    mesh_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh");
    mesh_prgs[1] = pipeline.createHitgroupProgram(context, module, "__closesthit__shadow");

    std::array<ProgramGroup, NRay> sphere_prgs;
    sphere_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__sphere");
    sphere_prgs[1] = pipeline.createHitgroupProgram(context, module, "__closesthit__shadow", "__intersection__sphere");

    std::array<ProgramGroup, NRay> plane_prgs;
    plane_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__plane");
    plane_prgs[1] = pipeline.createHitgroupProgram(context, module, "__closesthit__shadow", "__intersection__plane");

    // Textures
    auto green_constant = make_shared<ConstantTexture>(Vec3f(0.05f, 0.8f, 0.05f), constant_prg.ID);
    auto red_constant = make_shared<ConstantTexture>(Vec3f(0.8f, 0.05f, 0.05f), constant_prg.ID);
    auto white_constant = make_shared<ConstantTexture>(Vec3f(0.8f), constant_prg.ID);
    auto floor_checker = make_shared<CheckerTexture>(Vec3f(0.3f), Vec3f(0.8f), 10, checker_prg.ID);

    // Materials
    auto green_diffuse = make_shared<Diffuse>(diffuse_id, green_constant);
    auto red_diffuse = make_shared<Diffuse>(diffuse_id, red_constant);
    auto white_diffuse = make_shared<Diffuse>(diffuse_id, white_constant);
    auto floor_diffuse = make_shared<Diffuse>(diffuse_id, floor_checker);

    // Objects
    auto wall_plane = make_shared<Plane>(Vec2f(-25.0f), Vec2f(25.0f));
    scene.addObject("left_wall", wall_plane, green_diffuse, plane_prgs,
        Matrix4f::translate(-25, 0, 0) * Matrix4f::rotate(math::pi / 2.0f, Vec3f{0, 0, 1}));

    
    scene.addObject("right_wall", wall_plane, red_diffuse, plane_prgs, 
        Matrix4f::translate(25, 0, 0)* Matrix4f::rotate(math::pi / 2.0f, Vec3f{ 0, 0, 1 }));

    scene.addObject("back_wall", wall_plane, white_diffuse, plane_prgs,
        Matrix4f::translate(0, 0, -25)* Matrix4f::rotate(math::pi / 2.0f, Vec3f{ 1, 0, 0 }));

    scene.addObject("ceiling", wall_plane, white_diffuse, plane_prgs,
        Matrix4f::translate(0, 25, 0));

    scene.addObject("floor", wall_plane, floor_diffuse, plane_prgs,
        Matrix4f::translate(0, -25, 0));

    scene.addLightObject("ceiling_light", make_shared<Plane>(Vec2f(-5.0f), Vec2f(5.0f)),
        make_shared<AreaEmitter>(area_emitter_id, make_shared<ConstantTexture>(Vec3f(1.0f), constant_prg.ID)), 
        plane_prgs, Matrix4f::translate(0, 24.9f, 0));

    CUDA_CHECK(cudaStreamCreate(&stream));
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

    pgSetWindowName(toString(pgGetFrameRate()));
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



