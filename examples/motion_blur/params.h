#pragma once 

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/optix/sbt.h>

namespace prayground {

struct Light 
{
    float3 pos;
};

struct LaunchParams 
{
    unsigned int width, height;
    uchar4* result_buffer;
    float4* accum_buffer;
    unsigned int subframe_index;
    unsigned int samples_per_launch;

    Light light;

    OptixTraversableHandle handle;
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

struct TextureData
{
    void* data; 
    unsigned int prg_id;
};

struct HitgroupData
{
    void* shape_data;
    TextureData tex_data;
};

struct MissData
{
    void* env_data;
};

struct EmptyData
{

};

} // ::prayground