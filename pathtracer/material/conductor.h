#pragma once

#include <include/core/material.h>
#include <include/core/bsdf.h>

namespace pt {

struct ConductorData {
    float3 albedo; 
    float fuzz;
};

#ifndef __CUDACC__
class Conductor final : public Material {
public:
    Conductor(float3 a, float f) 
    : m_albedo(a), m_fuzz(f) {}
    ~Conductor() {}

    void sample( SurfaceInteraction& si ) const override {
        si.wo = reflect(si.wi, si.n);
        si.attenuation = m_albedo;
        si.trace_terminate = false;
    }
    
    float3 emittance( SurfaceInteraction& si ) const override { return make_float3(0.f); }  

    void prepare_data() override {
        ConductorData data = {
            m_albedo, 
            m_fuzz
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
    float3 m_albedo;
    float m_fuzz;
};

#else 
CALLABLE_FUNC void DC_FUNC(sample_conductor)(SurfaceInteraction* si, void* matdata) {
    const ConductorData* conductor = reinterpret_cast<ConductorData*>(matdata);

    si->wo = reflect(si->wi, si->n);
    si->attenuation = conductor->albedo;
    si->trace_terminate = false;
}

#endif

}



