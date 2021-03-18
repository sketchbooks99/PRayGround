#pragma once
#include <include/core/texture.h>

namespace pt {

class CheckerTexture final : public Texture {
public:
    HOSTDEVICE CheckerTexture(const float3& c1, const float3& c2, float s=5)
    : m_color1(c1), m_color2(c2), m_scale(s) {
        #ifndef __CUDACC__
        setup_on_device();
        #endif
    }
    HOSTDEVICE ~CheckerTexture() {
        #ifndef __CUDACC__
        delete_on_device();
        #endif
    }
    HOSTDEVICE float3 eval(const SurfaceInteraction& si) const override { 
        const bool is_odd = sinf(si.uv.x*M_PIf*0) * sin(si.uv.y*M_PIf*m_scale) < 0;
        return is_odd ? c1 : c2;
    }
    HOST TextureType type() const override { return TextureType::Checker; }
private:
    HOST setup_on_device() override {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ptr), sizeof(TexturePtr));
        create_object_on_device<<<1,1>>>(reinterpret_cast<CheckerTexture**>(&d_ptr), m_color1, m_color2, m_scale);
        CUDA_SYNC_CHECK();
    }
    HOST delete_on_device() override {
        delete_object_on_device<<<1,1>>>(reinterpret_cast<CheckerTexture**>(&d_ptr));
        CUDA_SYNC_CHECK();
    }
    float3 m_color1, m_color2;
    float m_scale;
}; 

}