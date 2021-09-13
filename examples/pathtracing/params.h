#pragma once 

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/interaction.h>

struct Triangle
{
    float3 v0, v1, v2;
};

struct AreaEmitterInfo
{
    // 面光源がメッシュの場合はMeshDataではなくTriangleになる
    void* shape_data;
    Matrix4f objToWorld;

    unsigned int sample_id;
    unsigned int pdf_id;
};

struct LaunchParams 
{
    unsigned int width, height;
    unsigned int samples_per_launch;
    unsigned int max_depth;
    int subframe_index;
    uchar4* result_buffer;
    float4* accum_buffer;
    OptixTraversableHandle handle;

    AreaEmitterInfo* lights;
    int num_lights;
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
    SurfaceInfo surface_info;
};

struct MissData
{
    void* env_data;
};

struct EmptyData
{

};