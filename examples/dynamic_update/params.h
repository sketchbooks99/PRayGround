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
    float3* normal_buffer;
    float3* albedo_buffer;

    Light light;

    OptixTraversableHandle handle;
};

struct CameraData
{
    float3 origin; 
    float3 lookat; 
    float3 up;
    float fov;
    float aspect;
    float nearclip; 
    float farclip;
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