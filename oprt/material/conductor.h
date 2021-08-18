#pragma once

#include <oprt/core/material.h>
#include <oprt/texture/constant.h>

namespace oprt {

struct ConductorData {
    void* texdata;
    float fuzz;
    bool twosided;
    uint32_t tex_func_id;
};

#ifndef __CUDACC__
class Conductor final : public Material {
public:
    Conductor(const float3& a, float f, bool twosided=true) 
    : m_texture(std::make_shared<ConstantTexture>(a)), m_fuzz(f), m_twosided(twosided) {}

    Conductor(const std::shared_ptr<Texture>& texture, float f, bool twosided=true)
    : m_texture(texture), m_fuzz(f), m_twosided(twosided) {}
    
    ~Conductor() {}

    void prepareData() override {
        if (!m_texture->devicePtr())
            m_texture->prepareData();

        ConductorData data = {
            m_texture->devicePtr(), 
            m_fuzz,
            m_twosided,
            m_texture->programId()
        };

        CUDA_CHECK(cudaMalloc(&d_data, sizeof(ConductorData)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(ConductorData), 
            cudaMemcpyHostToDevice
        ));
    }

    void freeData() override
    {
        m_texture->freeData();
    }

    MaterialType type() const override { return MaterialType::Conductor; }    
private:
    std::shared_ptr<Texture> m_texture;
    float m_fuzz;
    bool m_twosided;
};

#endif

}



