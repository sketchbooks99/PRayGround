#pragma once
#include <prayground/core/texture.h>

namespace prayground {

struct CheckerTextureData {
    float3 color1;
    float3 color2;
    float scale;
};

#ifndef __CUDACC__
class CheckerTexture final : public Texture {
public:
    CheckerTexture(const float3& c1, const float3& c2, float s=5)
    : m_color1(c1), m_color2(c2), m_scale(s) {}
    ~CheckerTexture() noexcept {}

    void copyToDevice() override {
        CheckerTextureData data = {
            m_color1, 
            m_color2, 
            m_scale
        };

        CUDA_CHECK(cudaMalloc(&d_data, sizeof(CheckerTextureData)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(CheckerTextureData),
            cudaMemcpyHostToDevice
        ));
    }

private:
    float3 m_color1, m_color2;
    float m_scale;
}; 

#endif // __CUDACC__

}