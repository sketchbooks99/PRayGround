#include "util.cuh"

extern "C" __device__ Vec4f __direct_callable__bitmap(const Vec2f& uv, void* tex_data) {
    const BitmapTexture::Data* image = reinterpret_cast<BitmapTexture::Data*>(tex_data);
    float4 c = tex2D<float4>(image->texture, uv.x(), uv.y());
    return c;
}

extern "C" __device__ Vec4f __direct_callable__constant(const Vec2f& uv, void* tex_data) {
    const ConstantTexture::Data* constant = reinterpret_cast<ConstantTexture::Data*>(tex_data);
    return constant->color;
}

extern "C" __device__ Vec4f __direct_callable__checker(const Vec2f& uv, void* tex_data) {
    const CheckerTexture::Data* checker = reinterpret_cast<CheckerTexture::Data*>(tex_data);
    const bool is_odd = sinf(uv.x() * math::pi * checker->scale) * sinf(uv.y() * math::pi * checker->scale) < 0;
    return lerp(checker->color1, checker->color2, (float)is_odd);
}