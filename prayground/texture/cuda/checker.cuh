#pragma once 

#include <prayground/optix/cuda/util.cuh>
#include "../checker.h"

CALLABLE_FUNC float3 DC_FUNC(eval_checker)(SurfaceInteraction* si, void* texdata) {
    const CheckerTextureData* checker = reinterpret_cast<CheckerTextureData*>(texdata);
    const bool is_odd = sinf(si->uv.x*math::pi*checker->scale) * sinf(si->uv.y*math::pi*checker->scale) < 0;
    return is_odd ? checker->color1 : checker->color2;
}