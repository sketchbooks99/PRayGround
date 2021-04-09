#pragma once

#include <cuda/random.h>
#include <include/core/material.h>
#include <include/core/bsdf.h>
#include <include/core/onb.h>
#include <include/optix/sbt.h>

namespace pt {

struct DiffuseData {
    float3 albedo;
};

#ifndef __CUDACC__

class Diffuse final : public Material {
public:
    explicit Diffuse(float3 a) : m_albedo(a) { }

    ~Diffuse() { }

    void sample(SurfaceInteraction& si) const override {
        unsigned int seed = si.seed;
        si.trace_terminate = false;

        {
            const float z1 = rnd(seed);
            const float z2 = rnd(seed);

            float3 w_in; 
            cosine_sample_hemisphere(z1, z2, w_in);
            Onb onb(si.n);
            onb.inverse_transform(w_in);
            si.wo = w_in;
        }

        si.seed = seed;
        si.attenuation = m_albedo;
    }
    
    float3 emittance(SurfaceInteraction& /* si */) const override { return make_float3(0.f); }

    float3 albedo() const { return m_albedo; }

    void prepare_data() override {
        DiffuseData data {
            m_albedo
        };

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(DiffuseData)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_data),
            &data, sizeof(DiffuseData), 
            cudaMemcpyHostToDevice
        ));
    }

    MaterialType type() const override { return MaterialType::Diffuse; }

private:
    float3 m_albedo;
};

#else
/**
 * \brief Sample bsdf at the surface.
 * \note This is direct callables function on the device, 
 *       so this function cannot launch `optixTrace()`.  
 */
CALLABLE_FUNC void DC_FUNC(sample_diffuse)(SurfaceInteraction* si, void* matdata) {
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(matdata);

    unsigned int seed = si->seed;
    si->trace_terminate = false;

    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in; 
        cosine_sample_hemisphere(z1, z2, w_in);
        Onb onb(si->n);
        onb.inverse_transform(w_in);
        si->wo = w_in;
    }
    si->seed = seed;
    si->attenuation = diffuse->albedo;
    si->emission = make_float3(0.0f);
}

#endif

}