#include "app.h"

// --------------------------------------------------------------------
void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocateDevicePtr();
    normal_bitmap.allocateDevicePtr();
    albedo_bitmap.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<float4*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<float4*>(accum_bitmap.devicePtr());
    params.normal_buffer = reinterpret_cast<float4*>(normal_bitmap.devicePtr());
    params.albedo_buffer = reinterpret_cast<float4*>(albedo_bitmap.devicePtr());

    CUDA_SYNC_CHECK();
}

// --------------------------------------------------------------------
void App::handleCameraUpdate()
{
    if (!camera_update)
        return;
    camera_update = false;

    float3 U, V, W;
    camera.UVWFrame(U, V, W);

    RaygenRecord* rg_record = reinterpret_cast<RaygenRecord*>(sbt.raygenRecord());
    RaygenData rg_data;
    rg_data.camera = 
    {
        .origin = camera.origin(),
        .lookat = camera.lookat(),
        .U = U,
        .V = V,
        .W = W,
        .nearclip = camera.nearClip(),
        .farclip = camera.farClip()
    };
}

// --------------------------------------------------------------------
void App::setup()
{
    // Initialize CUDA
    stream = 0;
    CUDA_CHECK(cudaFree(0));
    // Initialize OptixDeviceContext
    OPTIX_CHECK(optixInit());
    context.disableValidation();
    context.create();

    // Initialize instance acceleration structure
}

// --------------------------------------------------------------------
void App::update()
{

}

// --------------------------------------------------------------------
void App::draw()
{

}

void App::close()
{

}

// --------------------------------------------------------------------
void App::keyPressed(int key)
{

}


