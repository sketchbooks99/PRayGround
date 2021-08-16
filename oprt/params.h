#pragma once 

#include <sutil/vec_math.h>
#include <optix.h>
#include "optix/sbt.h"

namespace oprt {

struct LaunchParams 
{
    uint32_t width, height;
    uint32_t samples_per_launch;
    uchar4* result_buffer;
    OptixTraversableHandle handle;
};

struct CameraData
{
    float3 origin; 
    float3 lookat; 
    float3 up;
    float fov;
};

struct RaygenData
{
    CameraData camera;
};

struct HitgroupData
{
    void* shape_data;
    void* surface_data;

    unsigned int surface_callable_func_id;   
};

struct MissData
{
    void* env_data;
};

using RaygenRecord = Record<RaygenData>;
using HitgroupRecord = Record<HitgroupData>;
using MissRecord = Record<MissData>;

} // ::oprt