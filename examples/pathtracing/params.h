#pragma once 

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/math/matrix.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/interaction.h>

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec3f>;
using CheckerTexture = CheckerTexture_<Vec3f>;

struct AreaEmitterInfo
{
    void* shape_data;
    Matrix4f objToWorld;
    Matrix4f worldToObj;

    uint32_t sample_id;
    uint32_t pdf_id;
    bool twosided = false;

    SurfaceInfo surface_info;
};

struct LaunchParams 
{
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_launch;
    uint32_t max_depth;
    int frame;
    Vec4u* result_buffer;
    Vec4f* accum_buffer;
    Vec3f* normal_buffer;
    Vec3f* albedo_buffer;
    float* depth_buffer;
    OptixTraversableHandle handle;

    AreaEmitterInfo* lights;
    int num_lights;

    float white;
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