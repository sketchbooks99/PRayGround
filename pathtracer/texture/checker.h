#pragma once
#include "../core/texture.h"

namespace pt {

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
    ~CheckerTexture(){}

    float3 eval(const SurfaceInteraction& si) const override { 
        const bool is_odd = sinf(si.uv.x*M_PIf*0) * sin(si.uv.y*M_PIf*m_scale) < 0;
        return is_odd ? m_color1 : m_color2;
    }

    void prepare_data() override {
        CheckerTextureData data = {
            m_color1, 
            m_color2, 
            m_scale
        };

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(CheckerTextureData)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_data),
            &data, sizeof(CheckerTextureData),
            cudaMemcpyHostToDevice
        ));
    }

    TextureType type() const override { return TextureType::Checker; }
private:
    float3 m_color1, m_color2;
    float m_scale;
}; 

#else

CALLABLE_FUNC float3 DC_FUNC(eval_checker)(SurfaceInteraction* si, void* texdata) {
    const CheckerTextureData* checker = reinterpret_cast<CheckerTextureData*>(texdata);
    const bool is_odd = sinf(si->uv.x*M_PIf*checker->scale) * sinf(si->uv.y*M_PIf*checker->scale) < 0;
    return is_odd ? checker->color1 : checker->color2;
}

#endif

}