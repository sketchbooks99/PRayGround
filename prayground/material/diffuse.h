#pragma once

#include <cuda/random.h>
#include <prayground/core/material.h>
#include <prayground/core/bsdf.h>
#include <prayground/core/onb.h>
#include <prayground/optix/sbt.h>
#include <prayground/texture/constant.h>

namespace prayground {

struct DiffuseData {
    void* tex_data;
    bool twosided;
    unsigned int tex_program_id;
};

#ifndef __CUDACC__

class Diffuse final : public Material {
public:
    explicit Diffuse(const std::shared_ptr<Texture>& texture, bool twosided=true)
    : m_texture(texture), m_twosided(twosided) {}

    ~Diffuse() { }

    void copyToDevice() override {
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();

        DiffuseData data {
            .tex_data = m_texture->devicePtr(),
            .twosided = m_twosided,
            .tex_program_id = m_texture->programId()
        };

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(DiffuseData)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(DiffuseData), 
            cudaMemcpyHostToDevice
        ));
    }

    void free() override
    {
        m_texture->free();
    }
private:
    std::shared_ptr<Texture> m_texture;
    bool m_twosided;
};

#else

#endif

}