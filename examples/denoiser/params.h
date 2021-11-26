#pragma once 

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/optix/sbt.h>
#include <prayground/optix/cuda/device_util.cuh>
#include <prayground/core/interaction.h>
#include <prayground/math/matrix.h>

using namespace prayground;

struct AreaEmitterInfo 
{
    void* shape_data;
    Matrix4f objToWorld;
    Matrix4f worldToObj;

    uint32_t sample_id; 
    uint32_t pdf_id; 
};

struct LaunchParams 
{
    unsigned int width, height;
    unsigned int samples_per_launch;
    unsigned int max_depth;
    int frame;
    float4* result_buffer;
    float4* accum_buffer;
    // For denoiser
    float4* albedo_buffer;
    float4* normal_buffer;
    OptixTraversableHandle handle;

    AreaEmitterInfo* lights;
    uint32_t num_lights;

    float white;
};

struct CameraData 
{
    float3 origin; 
    float3 lookat; 
    float3 U; 
    float3 V; 
    float3 W;
    float farclip;
};

struct RaygenData
{
    CameraData camera;
};

struct HitgroupData
{
    void* shape_data;
    SurfaceInfo surface_info;
};

struct MissData
{
    void* env_data;
};

struct EmptyData
{

};