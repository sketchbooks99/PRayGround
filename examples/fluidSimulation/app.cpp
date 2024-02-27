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
    is_camera_udpate = true;

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
    pipeline.setAttributes(5);

    // Create module
    Module module = pipeline.createModuleFromCudaFile(context, "kernels.cu");

    const int width = pgGetWidth();
    const int height = pgGetHeight();
    result_bmp.allocate(PixelFormat::RGBA, width, height);
    result_bmp.allocateDevicePtr();
    accum_bmp.allocate(PixelFormat::RGBA32F, width, height);
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
    ProgramGroup raygen = pipeline.createRaygenProgramGroup(context, module, "__raygen__pinhole");
    pipeline.setRaygen(raygen);

    auto setupCallable = [&](const string& dc_name, const string& cc_name) {
        Callable callable = pipeline.createCallableProgram(context, module, dc_name, cc_name);
        scene.bindCallablesProgram(callable.program);
        return callable.ID;
    };

    // Texture programs
    auto bitmap_id = setupCallable("__direct_callable__bitmap", "");
    auto checker_id = setupCallable("__direct_callable__checker", "");
    auto constant_id = setupCallable("__direct_callable__constant", "");

    // Miss program
    ProgramGroup miss = pipeline.createMissProgramGroup(context, module, "__miss__envmap");
    scene.bindMissPrograms({miss});
    auto envmap_texture = make_shared<FloatBitmapTexture>("resources/image/drackenstein_quarry_4k.exr", bitmap_id);
    scene.setEnvmap(envmap_texture);

    // Hitgroup program
    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh");
    ProgramGroup sphere_prg = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__sphere");
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



