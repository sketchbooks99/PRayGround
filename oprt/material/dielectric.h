#pragma once 

#include <cuda/random.h>
#include <oprt/core/material.h>
#include <oprt/core/bsdf.h>
#include <oprt/texture/constant.h>

namespace oprt {

struct DielectricData {
    void* tex_data;
    float ior;
    unsigned int tex_program_id;
};

#ifndef __CUDACC__

class Dielectric final : public Material {
public:
    Dielectric(const std::shared_ptr<Texture>& texture, float ior)
    : m_texture(texture), m_ior(ior) {}

    ~Dielectric() { }

    void copyToDevice() override {
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();

        DielectricData data = {
            m_texture->devicePtr(), 
            m_ior, 
            m_texture->programId()
        };

        CUDA_CHECK(cudaMalloc(&d_data, sizeof(DielectricData)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(DielectricData),
            cudaMemcpyHostToDevice
        ));
    }

    void free() override
    {
        m_texture->free();
    }

    MaterialType type() const override { return MaterialType::Dielectric; }

private:
    // float3 m_albedo;
    std::shared_ptr<Texture> m_texture;
    float m_ior;
};

#else 

#endif

}