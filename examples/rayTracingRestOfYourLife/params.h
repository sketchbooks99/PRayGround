#pragma once 

#include <optix.h>
#include <prayground/math/vec.h>
#include <prayground/math/matrix.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/camera.h>
#include <prayground/core/interaction.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec3f>;
using CheckerTexture = CheckerTexture_<Vec3f>;

enum class RayType : uint32_t
{
    RADIANCE = 0,
    SHADOW = 1,
    N_RAY = 2
};

struct AreaEmitterInfo
{
    void* shape_data;
    SurfaceInfo surface_info;

    Matrix4f objToWorld;
    Matrix4f worldToObj;

    uint32_t sample_id;
    uint32_t pdf_id;
};

struct LightInteraction
{
    // A surface point on the light source in world coordinates
    Vec3f p;
    // Surface normal on the light source in world coordinates
    Vec3f n;
    // Texture coordinates on light source
    Vec2f uv;
    // Area of light source
    float area;
    // PDF of light source
    float pdf;
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