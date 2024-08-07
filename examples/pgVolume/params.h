#pragma once 

#include <optix.h>
#include <prayground/core/camera.h>
#include <prayground/math/vec_math.h>
#include <prayground/math/matrix.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/interaction.h>

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec3f>;
using CheckerTexture = CheckerTexture_<Vec3f>;
using VDBGrid = VDBGrid_<Vec3f>;

struct LaunchParams 
{
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;
    unsigned int max_depth;
    int frame;
    Vec4u* result_buffer;
    Vec4f* accum_buffer;
    OptixTraversableHandle handle;

    float cloud_opacity;
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
