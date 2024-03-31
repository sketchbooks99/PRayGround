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
    is_camera_updated = true;

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

    // Camera settings
    std::shared_ptr<Camera> camera = make_shared<Camera>();
    camera->setOrigin(0, 0, -100);
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

    // Miss program
    ProgramGroup miss = pipeline.createMissProgram(context, module, "__miss__envmap");
    scene.bindMissPrograms({miss});
    auto envmap_texture = make_shared<FloatBitmapTexture>("resources/image/drackenstein_quarry_4k.exr", bitmap_id);
    scene.setEnvmap(envmap_texture);

    // Hitgroup program
    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh");
    ProgramGroup particle_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__particle");

    // Surface programs
    SurfaceCallableID diffuse_id = {
        .sample = setupCallable("__direct_callable__sample_diffuse", ""),
        .bsdf = setupCallable("__direct_callable__bsdf_diffuse", ""),
        .pdf = setupCallable("__direct_callable__pdf_diffuse", "")
    };
    auto area_emitter_callable_id = setupCallable("__direct_callable__area_emitter", "");
    SurfaceCallableID area_emitter_id = {
        .sample = area_emitter_callable_id,
        .bsdf = area_emitter_callable_id,
        .pdf = area_emitter_callable_id
    };

    // Create surfaces
    auto floor = make_shared<Diffuse>(diffuse_id, make_shared<CheckerTexture>(Vec3f(0.2f), Vec3f(0.8f), 10, checker_id));
    auto particle = make_shared<Diffuse>(diffuse_id, make_shared<ConstantTexture>(Vec3f(0.3f, 0.3f, 0.9f), constant_id));

    // Initialize fluid particles
    float radius = 1.0f;
    particles = make_shared<ShapeGroup<SPHParticle, ShapeType::Custom>>();
    for (int x = 0; x < 50; x++) {
        for (int y = 0; y < 50; y++) {
            for (int z = 0; z < 50; z++) {
                Vec3f position = Vec3f(x, y, z) * 2.0f;
                Vec3f velocity = Vec3f(0);
                float mass = 1.0f;
                auto p = SPHParticle(position, velocity, mass, radius);
                particles->addShape(p);
            }
        }
    }

    // Fluid particle
    scene.addObject("particles", particles, particle, { particle_prg }, Matrix4f::identity());

    // Floor
    scene.addObject("floor", make_shared<Plane>(Vec2f(-100), Vec2f(100)), floor, { mesh_prg }, Matrix4f::identity());

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
        .time_step = 0.01f,
        .stiffness = 1000.0f
    };

    params.sph_config = sph_config;
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



