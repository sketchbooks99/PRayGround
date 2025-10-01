#pragma once

#include <optix.h>
#include <prayground/math/vec.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>
#include "textures.h"

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec3f>;
using CheckerTexture = CheckerTexture_<Vec3f>;
using ProceduralWoodenTexture = ProceduralWoodenTexture_<Vec3f>;

struct LaunchParams {
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_launch;
    int32_t frame;
    uint32_t max_depth;
    
    // Elapsed time from the start of rendering (in seconds)
    float elapsed_time;

    Vec4u* result_buffer;
    Vec4f* accum_buffer;
    OptixTraversableHandle handle;
};

struct ProceduralTreeData {
    float gravity;
    float horizontal_strength;
    float scale;
    int depth;
};