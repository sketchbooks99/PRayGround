#pragma once
#include "../core/texture.h"

namespace pt {

/**
 * @note I have to carefully consider how to manage texture.
 */
struct ImageTextureData {
    /// @note Which is better?
    // float3* data;
    // CUtexObject* tex_object;
    float3 color;
};

#ifndef __CUDACC__
class ImageTexture final : public Texture {
public:
    ImageTexture(const float3& color) : m_color(color){}

    float3 eval(const SurfaceInteraction& si) const override { return make_float3(1.0f); }

    void prepare_data() override {
        ImageTextureData data = { m_color };

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(ImageTextureData)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_data), 
            &data, sizeof(ImageTextureData),
            cudaMemcpyHostToDevice
        ));
    }
    TextureType type() const override { return TextureType::Constant; }
private:
    float3 m_color;
};

#else
CALLABLE_FUNC float3 DC_FUNC(eval_image)(SurfaceInteraction* si, void* texdata) {
    const ImageTextureData* image = reinterpret_cast<ImageTextureData*>(texdata);
    return image->color;
}

#endif

}