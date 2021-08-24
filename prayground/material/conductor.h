#pragma once

#include <prayground/core/material.h>
#include <prayground/texture/constant.h>

namespace prayground {

struct ConductorData {
    void* tex_data;
    float fuzz;
    bool twosided;
    uint32_t tex_program_id;
};

#ifndef __CUDACC__
class Conductor final : public Material {
public:
    Conductor(const std::shared_ptr<Texture>& texture, float f, bool twosided=true)
    : m_texture(texture), m_fuzz(f), m_twosided(twosided) {}
    
    ~Conductor() {}

    void copyToDevice() override {
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();

        ConductorData data = {
            .tex_data = m_texture->devicePtr(), 
            .fuzz = m_fuzz,
            .twosided = m_twosided,
            .tex_program_id = m_texture->programId()
        };

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(ConductorData)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(ConductorData), 
            cudaMemcpyHostToDevice
        ));
    }

    void free() override
    {
        m_texture->free();
    }

private:
    std::shared_ptr<Texture> m_texture;
    float m_fuzz;
    bool m_twosided;
};

#endif

}



