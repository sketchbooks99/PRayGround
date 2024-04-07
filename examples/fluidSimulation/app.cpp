#include "app.h"

void App::initResultBufferOnDevice()
{
    params.frame = 0;
    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();

    params.result_buffer = (Vec4u*)result_bmp.deviceData();
    params.accum_buffer = (Vec4f*)accum_bmp.deviceData();
}

void App::handleCameraUpdate() {
    if (!is_camera_updated)
        return;
    is_camera_updated = false;

    scene.updateSBT(+(SBTRecordType::Raygen));

    initResultBufferOnDevice();
}

// ------------------------------------------------------------------
void App::setup()
{
    stream = 0;
    CUDA_CHECK(cudaFree(0));

    // Initialize context
    OPTIX_CHECK(optixInit());
    context.disableValidation();
    context.create();

    // Initialize pipeline
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(5);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);

    // Create module
    Module module = pipeline.createModuleFromCudaFile(context, "kernels.cu");

    const int width = pgGetWidth();
    const int height = pgGetHeight();
    result_bmp.allocate(PixelFormat::RGBA, width, height);
    result_bmp.allocateDevicePtr();
    accum_bmp.allocate(PixelFormat::RGBA, width, height);
    accum_bmp.allocateDevicePtr();

    // Configuration of launch parameters
    params.width = width;
    params.height = height;
    params.samples_per_launch = 1;
    params.frame = 0;
    params.max_depth = 5;
    params.result_buffer = (Vec4u*)result_bmp.deviceData();
    params.accum_buffer = (Vec4f*)accum_bmp.deviceData();

    scene.setup({ true, true });

    // Camera settings
    std::shared_ptr<Camera> camera = make_shared<Camera>();
    camera->setOrigin(300, 300, 300);
    camera->setLookat(0, 0, 0);
    camera->setUp(0, 1, 0);
    camera->setFov(40);
    camera->setAspect((float)width / height);
    camera->enableTracking(pgGetCurrentWindow());
    scene.setCamera(camera);

    // Raygen program
    ProgramGroup raygen = pipeline.createRaygenProgram(context, module, "__raygen__pinhole");
    scene.bindRaygenProgram(raygen);

    struct Callable {
        Callable(const std::pair<ProgramGroup, uint32_t>& callable)
            : program(callable.first), ID(callable.second) {}
        ProgramGroup program;
        uint32_t ID;
    };

    auto setupCallable = [&](const string& dc_name, const string& cc_name) {
        Callable callable = pipeline.createCallablesProgram(context, module, dc_name, cc_name);
        scene.bindCallablesProgram(callable.program);
        return callable.ID;
    };

    // Texture programs
    auto bitmap_id = setupCallable("__direct_callable__bitmap", "");
    auto checker_id = setupCallable("__direct_callable__checker", "");
    auto constant_id = setupCallable("__direct_callable__constant", "");

    pgLog(bitmap_id, checker_id, constant_id);

    // Miss program
    ProgramGroup miss = pipeline.createMissProgram(context, module, "__miss__envmap");
    scene.bindMissPrograms({miss});
    auto envmap_texture = make_shared<FloatBitmapTexture>("resources/image/drackenstein_quarry_4k.exr", bitmap_id);
    scene.setEnvmap(envmap_texture);

    // Hitgroup program
    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh");
    ProgramGroup plane_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__plane");
    ProgramGroup particle_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__particle");

    // Surface programs
    SurfaceCallableID diffuse_id = {
        .sample = setupCallable("__direct_callable__sample_diffuse", ""),
        .bsdf = setupCallable("__direct_callable__bsdf_diffuse", ""),
        .pdf = setupCallable("__direct_callable__pdf_diffuse", "")
    };
    SurfaceCallableID dielectric_id = {
        .sample = setupCallable("__direct_callable__sample_dielectric", ""),
        .bsdf = setupCallable("__direct_callable__bsdf_dielectric", ""),
        .pdf = setupCallable("__direct_callable__pdf_dielectric", "")
    };
    auto area_emitter_callable_id = setupCallable("__direct_callable__area_emitter", "");
    SurfaceCallableID area_emitter_id = {
        .sample = area_emitter_callable_id,
        .bsdf = area_emitter_callable_id,
        .pdf = area_emitter_callable_id
    };

    // Create surfaces
    auto floor_bsdf = make_shared<Diffuse>(diffuse_id, make_shared<CheckerTexture>(Vec3f(0.2f), Vec3f(0.8f), 10, checker_id));
    auto particle_bsdf = make_shared<Dielectric>(dielectric_id, make_shared<ConstantTexture>(Vec3f(0.8f), constant_id), 1.5f);

    // Initialize fluid particles
    float radius = 1.0f;
    std::vector<SPHParticles::Data> particle_data;
    for (int x = 0; x < 50; x++) {
        for (int y = 0; y < 50; y++) {
            for (int z = 0; z < 50; z++) {
                Vec3f position = Vec3f(x, y, z) * 3.0f - 75.0f;
                Vec3f velocity = Vec3f(0);
                float mass = 1.0f;
                auto p = SPHParticles::Data{ position, velocity, mass, radius, 0.0f, 0.0f, Vec3f(0.0f)};
                particle_data.push_back(p);
            }
        }
    }
    particles = make_shared<SPHParticles>(particle_data);

    // Fluid particle
    scene.addObject("particles", particles, particle_bsdf, { particle_prg }, Matrix4f::identity(), {true, true});

    // Floor
    scene.addObject("floor", make_shared<Plane>(Vec2f(-100), Vec2f(100)), floor_bsdf, { plane_prg }, Matrix4f::translate(0, -100, 0));

    CUDA_CHECK(cudaStreamCreate(&stream));
    scene.copyDataToDevice();
    scene.buildAccel(context, stream);
    scene.buildSBT();
    pipeline.create(context);

    params.handle = scene.accelHandle();

    // Configuration of SPH parameter
    sph_config = {
        .kernel_size = radius, 
        .rest_density = 1.0f,
        .external_force = Vec3f(0, -9.8f, 0),
        .time_step = 0.1f,
        .stiffness = 1000.0f
    };

    params.sph_config = sph_config;
}

// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();
    initResultBufferOnDevice();

    scene.launchRay(context, pipeline, params, stream, result_bmp.width(), result_bmp.height(), 1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    result_bmp.copyFromDevice();

    params.frame++;

    solveSPH((SPHParticles::Data*)particles->devicePtr(), particles->numPrimitives(), params.sph_config);

    scene.updateObjectGAS("particles", context, stream);
    scene.updateAccel(context, stream);
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



