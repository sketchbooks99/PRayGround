#include "util.cuh"
#include <prayground/texture/bitmap.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>

using namespace prayground;

extern "C" __device__ float4 __direct_callable__bitmap(const float2& uv, void* tex_data) {
    const BitmapTextureData* image = reinterpret_cast<BitmapTextureData*>(tex_data);
    float4 c = tex2D<float4>(image->texture, uv.x, uv.y);
    return c;
}

extern "C" __device__ float4 __direct_callable__constant(const float2& uv, void* tex_data) {
    const ConstantTextureData* constant = reinterpret_cast<ConstantTextureData*>(tex_data);
    return constant->color;
}

extern "C" __device__ float4 __direct_callable__checker(const float2& uv, void* tex_data) {
    const CheckerTextureData* checker = reinterpret_cast<CheckerTextureData*>(tex_data);
    const bool is_odd = sinf(uv.x*M_PIf*checker->scale) * sinf(uv.y*M_PIf*checker->scale) < 0;
    return is_odd ? checker->color1 : checker->color2;
}