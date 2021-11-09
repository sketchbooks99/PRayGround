#pragma once 

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/optix/sbt.h>

namespace prayground {

struct LaunchParams 
{
    unsigned int width, height;
    unsigned int samples_per_launch;
    unsigned int max_depth;
    int subframe_index;
    uchar4* result_buffer;
    OptixTraversableHandle handle;
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