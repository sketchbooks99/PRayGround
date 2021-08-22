#pragma once 

#include <oprt/optix/cuda/util.cuh>
#include "../bitmap.h"

CALLABLE_FUNC float3 DC_FUNC(eval_bitmap)(SurfaceInteraction* si, void* texdata) {
    const BitmapTextureData* image = reinterpret_cast<BitmapTextureData*>(texdata);
    float4 c = tex2D<float4>(image->texture, si->uv.x, si->uv.y);
    return make_float3(c.x, c.y, c.z);
}