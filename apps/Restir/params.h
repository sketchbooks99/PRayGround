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
    Radiance = 0,
    Shadow = 1,
    NRay = 2
};

struct Triangle {
    Vec3f v0;
    Vec3f v1;
    Vec3f v2;
    Vec3f n;
};

// Should be polygonal light source?
struct LightInfo {
    Triangle triangle;
    Vec3f emission;
};

struct LaunchParams {
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_launch;
    uint32_t max_depth;
    int frame;

    Vec4u* result_buffer;
    Vec4f* accum_buffer;
    
    OptixTraversableHandle handle;

    LightInfo* lights;
    int num_lights;
};