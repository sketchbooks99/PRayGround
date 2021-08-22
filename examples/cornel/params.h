#pragma once 

#include <sutil/vec_math.h>
#include <optix.h>
#include <oprt/optix/sbt.h>

namespace oprt {

struct LaunchParams 
{
    unsigned int width, height;
    unsigned int samples_per_launch;
    unsigned int max_depth;
    int subframe_index;
    uchar4* result_buffer;
    float4* accum_buffer;
    OptixTraversableHandle handle;
};

struct CameraData
{
    float3 origin; 
    float3 lookat; 
    float3 up;
    float fov;
    float aspect;
};

struct RaygenData
{
    CameraData camera;
};

struct HitgroupData
{
    void* shape_data;
    void* surface_data;

    unsigned int surface_program_id;   
    SurfaceType surface_type;
};

struct MissData
{
    void* env_data;
};

struct EmptyData
{

};

} // ::oprt