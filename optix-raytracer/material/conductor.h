#pragma once

#include "../core/material.h"
#include "../core/bsdf.h"
#include "../texture/constant.h"

namespace oprt {

struct ConductorData {
    void* texdata;
    float fuzz;
    bool twosided;
    unsigned int tex_func_idx;
};

#ifndef __CUDACC__
class Conductor final : public Material {
public:
    Conductor(const float3& a, float f, bool twosided=true) 
    : m_texture(new ConstantTexture(a)), m_fuzz(f), m_twosided(twosided) {}
    Conductor(Texture* texture, float f, bool twosided=true)
    : m_texture(texture), m_fuzz(f), m_twosided(twosided) {}
    ~Conductor() {}

    void prepare_data() override {
        m_texture->prepare_data();

        ConductorData data = {
            m_texture->get_dptr(), 
            m_fuzz,
            m_twosided,
            static_cast<unsigned int>(m_texture->type()) + static_cast<unsigned int>(MaterialType::Count) * 2
        };

        CUDA_CHECK(cudaMalloc(&d_data, sizeof(ConductorData)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(ConductorData), 
            cudaMemcpyHostToDevice
        ));
    }

    MaterialType type() const override { return MaterialType::Conductor; }
    
private:
    Texture* m_texture;
    float m_fuzz;
    bool m_twosided;
};

#else 

CALLABLE_FUNC void DC_FUNC(sample_conductor)(SurfaceInteraction* si, void* matdata) {
    const ConductorData* conductor = reinterpret_cast<ConductorData*>(matdata);
    if (conductor->twosided) 
        si->n = faceforward(si->n, -si->wi, si->n);

    si->wo = reflect(si->wi, si->n);
    si->trace_terminate = false;
    si->radiance_evaled = false;
}

CALLABLE_FUNC float3 CC_FUNC(bsdf_conductor)(SurfaceInteraction* si, void* matdata)
{
    const ConductorData* conductor = reinterpret_cast<ConductorData*>(matdata);
    si->emission = make_float3(0.0f);
    return optixDirectCall<float3, SurfaceInteraction*, void*>(conductor->tex_func_idx, si, conductor->texdata);
}

CALLABLE_FUNC float DC_FUNC(pdf_conductor)(SurfaceInteraction* si, void* matdata)
{
    return 1.0f;
}

#endif

}



