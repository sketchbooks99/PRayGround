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
    if (is_camera_updated)
        return;

    is_camera_updated = true;

    scene.updateSBT(+(SBTRecordType::Raygen));

    initResultBufferOnDevice();
}

// ------------------------------------------------------------------
void App::setup()
{
    // Initlaize CUDA
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

    // Initialize camera
    shared_ptr<Camera> camera(new Camera);
    camera->setOrigin(0, 0, 100);
    camera->setLookat(0, 0, 0);
    camera->setUp(0, 1, 0);
    camera->setFov(40);
    camera->setAspect((float)width / height);
    camera->enableTracking(pgGetCurrentWindow());
    scene.setCamera(camera);

    // Raygen program
    ProgramGroup rg_prg = pipeline.createRaygenProgram(context, module, "__raygen__restir");
    scene.bindRaygenProgram(rg_prg);

    struct Callables {
        Callables(const pair<ProgramGroup, uint32_t>& callable)
            : program(callable.first), ID(callable.second) {}
        ProgramGroup program;
        uint32_t ID;
    };
    Callables bitmap_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__bitmap", "");
    Callables constant_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__constant", "");
    Callables checker_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__checker", "");
    scene.bindCallablesProgram(bitmap_prg.program);
    scene.bindCallablesProgram(constant_prg.program);
    scene.bindCallablesProgram(checker_prg.program);

    // Create callables program for surfaces
    Callables diffuse_brdf_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__brdf_diffuse", "");
    Callables disney_brdf_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__brdf_disney", "");
    Callables area_emitter_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__area_emitter", "");
    scene.bindCallablesProgram(diffuse_brdf_prg.program);
    scene.bindCallablesProgram(disney_brdf_prg.program);
    scene.bindCallablesProgram(area_emitter_prg.program);

    // Miss program
    array<ProgramGroup, NRay> miss_prgs;
    miss_prgs[0] = pipeline.createMissProgram(context, module, "__miss__envmap");
    miss_prgs[1] = pipeline.createMissProgram(context, module, "__miss__shadow");
    scene.bindMissPrograms(miss_prgs);
    scene.setEnvmap(make_shared<ConstantTexture>(Vec3f(0.0f), constant_prg.ID));

    // Hitgroup program
    array<ProgramGroup, NRay> mesh_prgs;
    mesh_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh");
    mesh_prgs[1] = pipeline.createHitgroupProgram(context, module, "__closesthit__shadow");


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



