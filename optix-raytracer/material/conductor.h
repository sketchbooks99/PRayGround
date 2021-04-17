#pragma once

#include "../core/material.h"
#include "../core/bsdf.h"
#include "../texture/constant.h"

namespace oprt {

struct ConductorData {
    void* texdata;
    float fuzz;
    unsigned int tex_func_idx;
};

#ifndef __CUDACC__
class Conductor final : public Material {
public:
    Conductor(const float3& a, float f) 
    : m_texture(new ConstantTexture(a)), m_fuzz(f) {}
    Conductor(Texture* texture, float f)
    : m_texture(texture), m_fuzz(f) {}
    ~Conductor() {}

    void prepare_data() override {
        m_texture->prepare_data();

        ConductorData data = {
            reinterpret_cast<void*>(m_texture->get_dptr()), 
            m_fuzz,
            static_cast<unsigned int>(m_texture->type()) + static_cast<unsigned int>(MaterialType::Count) 
        };

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(ConductorData)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_data),
            &data, sizeof(ConductorData), 
            cudaMemcpyHostToDevice
        ));
    }

    MaterialType type() const override { return MaterialType::Conductor; }
    
private:
    Texture* m_texture;
    float m_fuzz;
};

#else 

CALLABLE_FUNC void CC_FUNC(sample_conductor)(SurfaceInteraction* si, void* matdata) {
    const ConductorData* conductor = reinterpret_cast<ConductorData*>(matdata);

    si->wo = reflect(si->wi, si->n);
    si->attenuation *= optixDirectCall<float3, SurfaceInteraction*, void*>(conductor->tex_func_idx, si, conductor->texdata);
    si->trace_terminate = false;
    si->emission = make_float3(0.0f);
    si->radiance = make_float3(0.0f);
}

#endif

}



