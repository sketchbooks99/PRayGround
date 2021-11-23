#pragma once 

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/interaction.h>

using namespace prayground;

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
    
    float white;
};

struct CameraData 
{
    float3 origin; 
    float3 lookat; 
    float3 U; 
    float3 V; 
    float3 W;
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

using RaygenRecord = Record<RaygenData>;
using HitgroupRecord = Record<HitgroupData>;
using MissRecord = Record<MissData>;
using EmptyRecord = Record<EmptyData>;