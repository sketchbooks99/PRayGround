#pragma once 

#include <optix.h>
#include <prayground/math/vec.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/camera.h>

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec3f>;
using CheckerTexture = CheckerTexture_<Vec3f>;

struct Light 
{
    Vec3f pos;
    Vec3f color;
    float intensity;
};

struct LaunchParams 
{
    uint32_t width;
    uint32_t height;
    Vec4u* result_buffer;
    Vec4f* accum_buffer;
    uint32_t frame;
    uint32_t samples_per_launch;

    Light light;

    OptixTraversableHandle handle;
};

struct RaygenData
{
    Camera::Data camera;
};

struct HitgroupData
{
    void* shape_data;
    Texture::Data texture;
};

struct MissData
{
    void* env_data;
};

struct EmptyData
{

};
