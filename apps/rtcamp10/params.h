#pragma once 

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/math/matrix.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/interaction.h>

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec3f>;
using CheckerTexture = CheckerTexture_<Vec3f>;

struct Triangle {
    Vec3f v0;
    Vec3f v1;
    Vec3f v2;
    Vec3f n;
};

struct LightInfo
{
    void* shape_data;
    Matrix4f objToWorld;
    Matrix4f worldToObj;

    // Callable ID for sampling the light
    uint32_t sample_id;
    // Callable ID for evaluating the PDF of the light
    uint32_t pdf_id;
    bool twosided = false;

    SurfaceInfo* surface_info;
};

struct LaunchParams 
{
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_launch;
    uint32_t max_depth;

    int frame;

    Vec4f* result_buffer;
    Vec4f* accum_buffer;
    Vec4f* normal_buffer;
    Vec4f* albedo_buffer;

    OptixTraversableHandle handle;

    LightInfo* lights;
    int num_lights;

    float white;
};