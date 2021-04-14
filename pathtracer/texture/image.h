#pragma once

#include "../core/texture.h"

namespace pt {

struct ImageTextureData {
    cudaTextureObject_t texture;
};

#ifndef __CUDACC__
class ImageTexture final : public Texture {
public:
    explicit ImageTexture(const std::string& filename);

    float3 eval(const SurfaceInteraction& si) const override { return make_float3(1.0f); }

    void prepare_data() override;
    TextureType type() const override { return TextureType::Constant; }
private:
    unsigned int width, height;
    float4* data;

    cudaArray_t d_array;
};

#else
CALLABLE_FUNC float3 DC_FUNC(eval_image)(SurfaceInteraction* si, void* texdata) {
    const ImageTextureData* image = reinterpret_cast<ImageTextureData*>(texdata);
}

#endif

}