#pragma once

#include "../core/texture.h"

namespace pt {

struct ImageTextureData {
    cudaTextureObject_t texture;
};

enum ImageFormat {
    UNSIGNED_BYTE4,
    FLOAT4, 
    FLOAT3
};

#ifndef __CUDACC__
class ImageTexture final : public Texture {
public:
    explicit ImageTexture(const std::string& filename);

    float3 eval(const SurfaceInteraction& si) const override { return make_float3(1.0f); }

    void prepare_data() override;
    TextureType type() const override { return TextureType::Constant; }
private:
    int width, height;
    int channels;
    uchar4* data;
    ImageFormat format;

    void _init_texture_desc();

    cudaTextureDesc tex_desc {};
    cudaArray_t d_array { nullptr };
};

#else
CALLABLE_FUNC float3 DC_FUNC(eval_image)(SurfaceInteraction* si, void* texdata) {
    const ImageTextureData* image = reinterpret_cast<ImageTextureData*>(texdata);
    float4 mask = tex2D<float4>(image->texture, si->uv.x, si->uv.y);
    return make_float3(mask.x, mask.y, mask.z);
}

#endif

}