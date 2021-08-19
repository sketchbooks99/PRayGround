#pragma once

#include <cuda/random.h>
#include <oprt/core/material.h>
#include <oprt/core/bsdf.h>
#include <oprt/core/onb.h>
#include <oprt/optix/sbt.h>
#include <oprt/texture/constant.h>

namespace oprt {

struct DiffuseData {
    void* texdata;
    bool twosided;
    unsigned int tex_func_id;
};

#ifndef __CUDACC__

class Diffuse final : public Material {
public:
    explicit Diffuse(const float3& a, bool twosided=true)
    : m_texture(std::make_shared<ConstantTexture>(a)), m_twosided(twosided) { }

    explicit Diffuse(const std::shared_ptr<Texture>& texture, bool twosided=true)
    : m_texture(texture), m_twosided(twosided) {}

    ~Diffuse() { }

    void copyToDevice() override {
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();

        DiffuseData data {
            m_texture->devicePtr(),
            m_twosided,
            m_texture->programId()
        };

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

    MaterialType type() const override { return MaterialType::Diffuse; }

private:
    std::shared_ptr<Texture> m_texture;
    bool m_twosided;
};

#else

#endif

}