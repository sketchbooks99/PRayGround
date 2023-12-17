#pragma once 

#include <optix.h>
#include <prayground/math/vec.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec4f>;
using CheckerTexture = CheckerTexture_<Vec4f>;

struct AreaEmitterInfo
{
    void* shape;
    SurfaceInfo surface_info;

    Matrix4f objToWorld;
    Matrix4f worldToObj;
};

struct LaunchParams {
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_launch;
    int32_t frame;
    uint32_t max_depth;

    Vec4u* result_buffer;
    Vec4f* accum_buffer;
    OptixTraversableHandle handle;

    AreaEmitterInfo* lights;
    int32_t num_lights;
};

struct RaygenData {
    Camera::Data camera;
};

struct HitgroupData {
    void* shape_data;
    SurfaceInfo surface_info;
    Texture::Data opacity_texture;
};

struct MissData {
    void* env_data;
};

struct EmptyData {

};