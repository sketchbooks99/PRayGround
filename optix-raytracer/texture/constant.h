#pragma once

#include "../core/texture.h"

namespace oprt {

struct ConstantTextureData {
    float3 color;
};

#ifndef __CUDACC__
class ConstantTexture final : public Texture {
public:
    explicit ConstantTexture(const float3& c) : m_color(c) {}
    ~ConstantTexture() {}

    float3 eval(const SurfaceInteraction& si) const override { return m_color; }

    void prepare_data() override {
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