#include "app.h"

// ------------------------------------------------------------------
void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.devicePtr());

    CUDA_SYNC_CHECK();
}

// ------------------------------------------------------------------
void App::handleCameraUpdate()
{
    if (!camera_update)
        return;
    camera_update = false;

    pgRaygenRecord<Camera>* rg_record = reinterpret_cast<pgRaygenRecord<Camera>*>(sbt.raygenRecord());
    pgRaygenData<Camera> rg_data;
    rg_data.camera = camera.getData();

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(&rg_record->data), 
        &rg_data, sizeof(pgRaygenData<Camera>), 
        cudaMemcpyHostToDevice
    ));

    initResultBufferOnDevice();
}

// ------------------------------------------------------------------
void App::launchGenLightVertices()
{
    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

    optixLaunch(
        static_cast<OptixPipeline>(genlight_pipeline),
        stream,
        d_params.devicePtr(),
        sizeof(LaunchParams),
        &sbt.sbt(),
        params.width,
        params.height,
        1
    );

    // Fetch light vertices from device
    h_light_vertices = params.light_vertices;
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

    // Initialize instance accel structure
    ias = InstanceAccel{ InstanceAccel::Type::Instances };

    // Pipeline settings
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(5);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(2);
    
    genlight_pipeline.setCompileOptions(pipeline.compileOptions());
    genlight_pipeline.setLinkOptions(pipeline.linkOptions());

    // Module creation
    Module rg_module, ms_module, ch_module, tex_module, surf_module, light_sample_module;
    rg_module = pipeline.createModuleFromCudaFile(context, "cuda/raygen.cu");
    ms_module = pipeline.createModuleFromCudaFile(context, "cuda/miss.cu");
    ch_module = pipeline.createModuleFromCudaFile(context, "cuda/closesthit.cu");
    tex_module = pipeline.createModuleFromCudaFile(context, "cuda/textures.cu");
    surf_module = pipeline.createModuleFromCudaFile(context, "cuda/surfaces.cu");
    light_sample_module = pipeline.createModuleFromCudaFile(context, "cuda/light_sample.cu");

    // Initialize bitmap
    result_bmp.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());
    accum_bmp.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());

    // Configuration of launch parameters
    params.width = result_bmp.width();
    params.height = result_bmp.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;

    initResultBufferOnDevice();

    // Camera settings
    camera.setOrigin(0, 0, -100);
    camera.setLookat(0, 0, 0);
    camera.setUp(0, 1, 0);
    camera.setFarClip(5000);
    camera.setAspect(40.0f);
    camera.enableTracking(pgGetCurrentWindow());
    
    ProgramGroup rg_genlight_prg;
    rg_genlight_prg.createRaygen(context, rg_module, "__raygen__lightpath");
    pgRaygenRecord<Camera> rg_record;
    rg_genlight_prg.recordPackHeader(&rg_record);
    sbt.setRaygenRecord(rg_record);

    // Raygen program for camera path
    ProgramGroup rg_prg = pipeline.createRaygenProgram(context, rg_module, "__raygen__camerapath");

    auto setupCallable = [&](const Module& m, const string& dc, const string& cc) -> uint32_t
    {
        pgCallableRecord ca_record{};
        auto [prg, id] = pipeline.createCallablesProgram(context, m, dc, cc);
        prg.recordPackHeader(&ca_record);
        sbt.addCallablesRecord(ca_record);
        return id;
    };

    // Callable programs for textures
    uint32_t constant_prg_id = setupCallable(tex_module, "__direct_callable__constant", "");
    uint32_t checker_prg_id = setupCallable(tex_module, "__direct_callable__checker", "");
    uint32_t bitmap_prg_id = setupCallable(tex_module, "__direct_callable__bitmap", "");

    // Callable programs for surfaces
    // Diffuse
    uint32_t diffuse_sample_bsdf_prg_id = setupCallable(surf_module, "__direct_callable__sample_diffuse", "__continuation_callable__bsdf_diffuse");
    uint32_t diffuse_pdf_prg_id = setupCallable(surf_module, "__direct_callable__pdf_diffuse", "");
    // Conductor
    uint32_t conductor_sample_bsdf_prg_id = setupCallable(surf_module, "__direct_callable__sample_conductor", "__continuation_callable__bsdf_conductor");
    uint32_t conductor_pdf_prg_id = setupCallable(surf_module, "__direct_callable__pdf_conductor", "");
    // Dielectric
    uint32_t dielectric_sample_bsdf_prg_id = setupCallable(surf_module, "__direct_callable__sample_dielectric", "__continuation_callable__bsdf_dielectric");
    uint32_t dielectric_pdf_prg_id = setupCallable(surf_module, "__direct_callable__pdf_dielectric", "");
    // Disney
    uint32_t disney_sample_bsdf_prg_id = setupCallable(surf_module, "__direct_callable__sample_disney", "__continuation_callable__bsdf_disney");
    uint32_t disney_pdf_prg_id = setupCallable(surf_module, "__direct_callable__pdf_disney", "");
    // Area emitter
    uint32_t area_emitter_prg_id = setupCallable(surf_module, "__direct_callable__area_emitter", "");

    // Callable program for direct sampling of area emitter
    uint32_t plane_sample_pdf_prg_id = setupCallable(light_sample_module, "__direct_callable__sample_plane", "__continuation_callable__pdf_plane");
    uint32_t sphere_sample_pdf_prg_id = setupCallable(light_sample_module, "__direct_callable__sample_sphere", "__continuation_callable__pdf_sphere");

    // Environment emitter
    textures.emplace("black", new ConstantTexture(Vec3f(0.0f), constant_prg_id));
    env = EnvironmentEmitter{ textures.at("black") };
    env.copyToDevice();

    // Miss program
    ProgramGroup ms_prg = pipeline.createMissProgram(context, ms_module, "__miss__envmap");
    pgMissRecord ms_record;
    ms_prg.recordPackHeader(&ms_record);
    ms_record.data.env_data = env.devicePtr();
    sbt.setMissRecord(ms_record);
}

// ------------------------------------------------------------------
void App::update()
{

}

// ------------------------------------------------------------------
void App::draw()
{

}

// ------------------------------------------------------------------
void App::mousePressed(float x, float y, int button)
{
    
}

// ------------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    
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
    
}

// ------------------------------------------------------------------
void App::keyPressed(int key)
{

}

// ------------------------------------------------------------------
void App::keyReleased(int key)
{

}



