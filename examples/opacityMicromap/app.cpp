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

    // Initialize device context
    OPTIX_CHECK(optixInit());
    context.disableValidation();
    context.create();

    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(0);
    pipeline.setContinuationCallableDepth(0);
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
    params.width = width;
    params.height = height;
    params.samples_per_launch = 4;
    params.frame = 0;
    params.max_depth = 5;
    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.devicePtr());

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

    // Miss program
    std::array<ProgramGroup, NRay> miss_prg{pipeline.createMissProgram(context, module, "__miss__envmap")};
    miss_prg[0] = pipeline.createMissProgram(context, module, "__miss__envmap");
    scene.bindMissPrograms(miss_prg);
    scene.setEnvmap(make_shared<ConstantTexture>(Vec3f(0.3f), 0));

    // Hitgroup program
    std::array<ProgramGroup, NRay> mesh_prg{pipeline.createHitgroupProgram(context, module, "__closesthit__mesh")};
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



