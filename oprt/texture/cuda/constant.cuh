#pragma once 

#include <oprt/optix/cuda/util.cuh>
#include "../constant.h"

CALLABLE_FUNC float3 DC_FUNC(eval_constant)(SurfaceInteraction* si, void* texdata) {
    const ConstantTextureData* constant = reinterpret_cast<ConstantTextureData*>(texdata);
    return constant->color;
}