#pragma once

#include <include/core/texture.h>

namespace pt {

class ConstantTexture final : public Texture {
public:
    explicit ConstantTexture(const float3& c) : m_color(c) {
        #ifndef __CUDACC__
        setup_on_device();
        #endif
    }
    ~ConstantTexture() {
        #ifndef __CUDACC__
        delete_on_device();
        #endif
    }

    HOSTDEVICE float3 eval(const SurfaceInteraction& si) const override { return m_color; }
    HOST TextureType type() const override { return TextureType::Constant; }
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

    float3 m_color;
};

}