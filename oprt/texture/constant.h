#pragma once

#include <oprt/core/texture.h>

namespace oprt {

struct ConstantTextureData {
    float3 color;
};

#ifndef __CUDACC__
class ConstantTexture final : public Texture {
public:
    explicit ConstantTexture(const float3& c) : m_color(c) {}
    ~ConstantTexture() noexcept {}

    void prepareData() override {
        ConstantTextureData data = { m_color };

        CUDA_CHECK(cudaMalloc(&d_data, sizeof(ConstantTextureData)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(ConstantTextureData),
            cudaMemcpyHostToDevice
        ));
    }
    
    TextureType type() const override { return TextureType::Constant; }
private:
    float3 m_color;
};

#else
CALLABLE_FUNC float3 DC_FUNC(eval_constant)(SurfaceInteraction* si, void* texdata) {
    const ConstantTextureData* constant = reinterpret_cast<ConstantTextureData*>(texdata);
    return constant->color;
}

#endif

}