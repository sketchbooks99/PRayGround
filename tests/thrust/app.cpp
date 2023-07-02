#include "app.h"

// ------------------------------------------------------------------
void App::setup()
{
    stream = 0; 
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    context.setDeviceId(0);
    context.create();

    pipeline.setLaunchVariableName("params");

    Module module = pipeline.createModuleFromCudaFile(context, "kernels.cu");

    params.width = pgGetWidth(); 
    params.height = pgGetHeight();

    Camera camera;

    ProgramGroup rg_prg = pipeline.createRaygenProgram(context, module, "__raygen__thrust");
    pgRaygenRecord<Camera> rg_record;
    rg_prg.recordPackHeader(&rg_record);
    rg_record.data.camera = camera.getData();
    sbt.setRaygenRecord(rg_record);

    ProgramGroup ms_prg = pipeline.createMissProgram(context, module, "__miss__void");
    pgMissRecord ms_record;
    ms_prg.recordPackHeader(&ms_record);
    ms_record.data.env_data = nullptr;
    sbt.setMissRecord(ms_record);

    sbt.createOnDevice();
    pipeline.create(context);
    CUDA_CHECK(cudaStreamCreate(&stream));
    d_params.allocate(sizeof(LaunchParams));
}

// ------------------------------------------------------------------
void App::update()
{
    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

    OPTIX_CHECK(optixLaunch(
        (OptixPipeline)pipeline, 
        stream, 
        d_params.devicePtr(), 
        sizeof(LaunchParams), 
        &sbt.sbt(), 
        params.width, 
        params.height, 
        1
    ));

    CUDA_CHECK(cudaSetDevice(context.deviceId()));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    pgLog(params.d_vector.size());
    params.d_vector.clear();
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



