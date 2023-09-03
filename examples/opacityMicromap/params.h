﻿#pragma once 

#include <optix.h>
#include <prayground/math/vec.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec4f>;
using CheckerTexture = CheckerTexture_<Vec4f>;

static constexpr int CONSTANT_TEXTURE_PRG_ID = 0;
static constexpr int CHECKER_TEXTURE_PRG_ID = 1;
static constexpr int BITMAP_TEXTURE_PRG_ID = 2;

struct LaunchParams {
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_launch;
    int32_t frame;
    uint32_t max_depth;

    Vec4u* result_buffer;
    Vec4f* accum_buffer;
    OptixTraversableHandle handle;
};