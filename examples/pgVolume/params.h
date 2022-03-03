#pragma once 

#include <optix.h>
#include <prayground/core/camera.h>
#include <prayground/math/vec_math.h>
#include <prayground/math/matrix.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/interaction.h>

using namespace prayground;

using ConstantTexture = ConstantTexture_<float3>;
using CheckerTexture = CheckerTexture_<float3>;
using GridMedium = GridMedium_<float3>;

struct LaunchParams 
{
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;
    unsigned int max_depth;
    int frame;
    uchar4* result_buffer;
    float4* accum_buffer;
    OptixTraversableHandle handle;
};

struct RaygenData
{
    Camera::Data camera;
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
