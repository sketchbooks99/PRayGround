#pragma once 

#include <cuda/random.h>
#include <prayground/core/material.h>
#include <prayground/core/bsdf.h>
#include <prayground/texture/constant.h>

namespace prayground {

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
            .tex_data = m_texture->devicePtr(), 
            .ior = m_ior, 
            .tex_program_id = m_texture->programId()
        };

        if (!d_data) 
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(DielectricData)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(DielectricData),
            cudaMemcpyHostToDevice
        ));
    }

    void setIor(const float ior)
    {
        m_ior = ior;
    }

    float ior() const
    {
        return m_ior;
    }

    std::shared_ptr<Texture> texture() const
    {
        return m_texture;
    }

    void free() override
    {
        m_texture->free();
    }
private:
    std::shared_ptr<Texture> m_texture;
    float m_ior;
};

#else 

#endif

}