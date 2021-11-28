#include "util.cuh"
#include <prayground/texture/bitmap.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>

using namespace prayground;

extern "C" __device__ float3 __direct_callable__bitmap(SurfaceInteraction* si, void* tex_data) {
    const BitmapTextureData* image = reinterpret_cast<BitmapTextureData*>(tex_data);
    float4 c = tex2D<float4>(image->texture, si->uv.x, si->uv.y);
    return make_float3(c.x, c.y, c.z);
}

extern "C" __device__ float3 __direct_callable__constant(SurfaceInteraction* si, void* tex_data) {
    const ConstantTextureData* constant = reinterpret_cast<ConstantTextureData*>(tex_data);
    return make_float3(constant->color);
}

extern "C" __device__ float3 __direct_callable__checker(SurfaceInteraction* si, void* tex_data) {
    const CheckerTextureData* checker = reinterpret_cast<CheckerTextureData*>(tex_data);
    const bool is_odd = sinf(si->uv.x*math::pi*checker->scale) * sinf(si->uv.y*math::pi*checker->scale) < 0;
    return is_odd ? make_float3(checker->color1) : make_float3(checker->color2);
}