#include "app.h"

#define ANIMATION 0

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

void App::initParticles() {
    float particle_radius = 2.0f;
    uint32_t seed = tea<4>(0, 0);
    std::vector<SPHParticles::Data> particle_data;
    
    // Sphere initialization parameters
    const Vec3f sphere_center(0.0f, 0.0f, 0.0f);  // Center of the sphere
    const float sphere_radius = 75.0f;             // Radius of the sphere to fill
    const float particle_spacing = 2.5f;          // Spacing between particles
    const int max_attempts = 100000;               // Maximum attempts to place particles
    
    // Generate particles inside sphere using rejection sampling
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        // Generate random position inside sphere
        Vec3f random_dir = normalize(UniformSampler::get3D(seed) * 2.0f - 1.0f);
        float random_radius = cbrtf(UniformSampler::get1D(seed)) * sphere_radius; // Uniform distribution in volume
        Vec3f position = sphere_center + random_dir * random_radius;
        
        // Check if position is far enough from existing particles
        bool valid_position = true;
        for (const auto& existing : particle_data) {
            if (length(position - existing.position) < particle_spacing) {
                valid_position = false;
                break;
            }
        }
        
        if (valid_position) {
            Vec3f velocity = Vec3f(0.0f);
            float mass = 1.0f;
            
            // Add small perturbation for natural distribution
            Vec3f perturbation = (UniformSampler::get3D(seed) - 0.5f) * 0.5f;
            position += perturbation;
            
            auto p = SPHParticles::Data{ position, velocity, mass, particle_radius, 0.0f, 0.0f, Vec3f(0.0f) };
            particle_data.push_back(p);
        }
    }
    
    
    particles->setParticles(particle_data);
    particles->copyToDevice();
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
    pipeline.setMaxTraversableGraphDepth(3);

    // Create module
    Module module = pipeline.createModuleFromOptixIr(context, "fluidSimulation_generated_kernels.cu.optixir");

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
    params.max_depth = 30;
    params.result_buffer = (Vec4u*)result_bmp.deviceData();
    params.accum_buffer = (Vec4f*)accum_bmp.deviceData();

    AppScene::AccelSettings accel_settings;
    accel_settings.allow_accel_compaction = true;
    accel_settings.allow_accel_update = true;
    scene.setup(accel_settings);

    // Camera settings
    std::shared_ptr<Camera> camera = make_shared<Camera>();
    camera->setOrigin(300, 200, 300);
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
    ProgramGroup plane_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__plane");
    ProgramGroup particle_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__particle");
    ProgramGroup sphere_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__sphere");

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
    particles = make_shared<SPHParticles>();
    initParticles();

    // Fluid particle
    AppScene::AccelSettings particle_accel_settings;
    particle_accel_settings.allow_accel_compaction = true;
    particle_accel_settings.allow_accel_update = true;
    scene.addObject("particles", particles, particle_bsdf, { particle_prg }, Matrix4f::identity(), particle_accel_settings);

    // Floor
    scene.addObject("floor", make_shared<Plane>(Vec2f(-200), Vec2f(200)), floor_bsdf, { plane_prg }, Matrix4f::translate(0, -100, 0));

    CUDA_CHECK(cudaStreamCreate(&stream));
    scene.copyDataToDevice();
    scene.buildAccel(context, stream);
    scene.buildSBT();
    pipeline.create(context);

    params.handle = scene.accelHandle();

    // Configuration of SPH parameter
    sph_config = {
        // Basic parameters
        .kernel_size = 7.0f, 
        .rest_density = 30.0f,
        .external_force = Vec3f(0, -9.8f, 0),
        .time_step = 0.1f,                 // Smaller initial timestep
        .stiffness = 0.02f,
        .viscosity = 0.05f,                 // Increased for better cohesion
        
        .ks = 10.0f,
        .kd = 10.0f
    };


    wall = AABB(Vec3f(-100, -97, -100), Vec3f(100, 500, 100));

    // GUI setting
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    const char* glsl_version = "#version 150";
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(pgGetCurrentWindow()->windowPtr(), true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    if (ANIMATION) {
        constexpr int FPS = 60;
        constexpr int SECONDS = 10;
        constexpr int SPP = 2048;
        for (int i = 0; i < FPS * SECONDS; i++) {
            initResultBufferOnDevice();

            std::cout << "Rendering frame " << i << "..." << std::endl;

            for (int j = 0; j < SPP / params.samples_per_launch; j++) {
                std::cout << "Sample " << j << "/" << SPP / params.samples_per_launch << "\r" << std::flush;
                scene.launchRay(context, pipeline, params, stream, result_bmp.width(), result_bmp.height(), 1);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_SYNC_CHECK();
                params.frame++;
            }

            result_bmp.copyFromDevice();
            
            auto filename = std::format("frame_{:03d}.png", i);
            auto filepath = pgPathJoin(pgAppDir(), filename);
            result_bmp.write(filepath);

            // Update particle physics first
            solveSPH((SPHParticles::Data*)particles->devicePtr(), particles->numPrimitives(), sph_config, wall);

            // Update geometry and acceleration structure with new particle positions
            scene.updateObjectGAS("particles", context, stream);
            scene.updateAccel(context, stream);
        }

        pgExit();
    }
}

// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();
    initResultBufferOnDevice();

    // Then render with updated positions
    scene.launchRay(context, pipeline, params, stream, result_bmp.width(), result_bmp.height(), 1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    result_bmp.copyFromDevice();

    params.frame++;

    // Update particle physics first
    solveSPH((SPHParticles::Data*)particles->devicePtr(), particles->numPrimitives(), sph_config, wall);

    // Update geometry and acceleration structure with new particle positions
    scene.updateObjectGAS("particles", context, stream);
    scene.updateAccel(context, stream);
}

// ------------------------------------------------------------------
void App::draw()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Fluid Simulation");

    ImGui::SliderFloat("Kernel Size", &sph_config.kernel_size, 1.0f, 100.0f);
    ImGui::SliderFloat("Rest Density", &sph_config.rest_density, 0.1f, 100.0f);
    ImGui::SliderFloat("Time Step", &sph_config.time_step, 0.001f, 0.2f);
    ImGui::SliderFloat("Stiffness", &sph_config.stiffness, 0.0f, 1.0f);
    ImGui::SliderFloat("Viscosity", &sph_config.viscosity, 0.0f, 1.0f);
    ImGui::SliderFloat("Ks", &sph_config.ks, 1.0f, 50.0f);
    ImGui::SliderFloat("Kd", &sph_config.kd, 1.0f, 50.0f);
    if (ImGui::Button("Reset")) {
        initParticles();
    }

    ImGui::End();
    ImGui::Render();

    result_bmp.draw(0, 0);

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
    if (key == Key::S)
        result_bmp.write("screenshot.png");
}

// ------------------------------------------------------------------
void App::keyReleased(int key)
{

}



