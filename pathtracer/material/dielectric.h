#pragma once 

#include <cuda/random.h>
#include <include/core/material.h>
#include <include/core/bsdf.h>

namespace pt {

struct DielectricData {
    float3 albedo;
    float ior;
};

#ifndef __CUDACC__

class Dielectric final : public Material {
public:
    Dielectric(float3 a, float ior)
    : m_albedo(a), m_ior(ior) { }
    ~Dielectric() { }

    void sample( SurfaceInteraction& si ) const override {
        si.attenuation = m_albedo;
        si.trace_terminate = false;
        
        float ni = 1.0f; // air
        float nt = m_ior;  // ior specified 
        float cosine = dot(si.wi, si.n);
        bool into = cosine < 0;
        float3 outward_normal = into ? si.n : -si.n;

        if (!into) swap(ni, nt);

        cosine = fabs(cosine);
        float sine = sqrtf(1.0 - cosine*cosine);
        bool cannot_refract = (ni / nt) * sine > 1.0f;

        float reflect_prob = fr(cosine, ni, nt);

        if (cannot_refract || reflect_prob > rnd(si.seed))
            si.wo = reflect(si.wi, outward_normal);
        else    
            si.wo = refract(si.wi, outward_normal, cosine, ni, nt);
        si.emission = make_float3(0.0f);
    }
    
    float3 emittance( SurfaceInteraction& /* si */ ) const override { return make_float3(0.f); }

    void prepare_data() override {
        DielectricData data = {
            m_albedo, 
            m_ior
        };

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(DielectricData)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_data),
            &data, sizeof(DielectricData),
            cudaMemcpyHostToDevice
        ));
    }

    MaterialType type() const override { return MaterialType::Dielectric; }

private:
    float3 m_albedo;
    float m_ior;
};

#else 
CALLABLE_FUNC void DC_FUNC(sample_dielectric)(SurfaceInteraction* si, void* matdata) {
    const DielectricData* dielectric = reinterpret_cast<DielectricData*>(matdata);

    si->attenuation = dielectric->albedo;
    si->trace_terminate = false;
    
    float ni = 1.0f; // air
    float nt = dielectric->ior;  // ior specified 
    float cosine = dot(si->wi, si->n);
    bool into = cosine < 0;
    float3 outward_normal = into ? si->n : -si->n;

    if (!into) swap(ni, nt);

    cosine = fabs(cosine);
    float sine = sqrtf(1.0 - cosine*cosine);
    bool cannot_refract = (ni / nt) * sine > 1.0f;

    float reflect_prob = fr(cosine, ni, nt);

    if (cannot_refract || reflect_prob > rnd(si->seed))
        si->wo = reflect(si->wi, outward_normal);
    else    
        si->wo = refract(si->wi, outward_normal, cosine, ni, nt);
    si->emission = make_float3(0.0f);
}

#endif

}