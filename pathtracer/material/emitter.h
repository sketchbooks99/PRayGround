#pragma once 

#include <include/core/material.h>

namespace pt {

struct EmitterData {
    float3 color;
    float strength;
};

#ifndef __CUDACC__
class Emitter final : public Material {
public:
    Emitter(const float3& color, float strength = 1.0f) 
    : m_color(color), m_strength(strength) { }

    ~Emitter() { }

    void sample(SurfaceInteraction& si) const override {
        si.trace_terminate = true;
    }
    float3 emittance(SurfaceInteraction& si) const override {
        return m_color * m_strength;
    }

    float3 emitted() const { return m_color; }

    void prepare_data() override {
        EmitterData data = {
            m_color, 
            m_strength
        };

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(EmitterData)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_data), 
            &data, sizeof(EmitterData), 
            cudaMemcpyHostToDevice
        ));
    }

    MaterialType type() const override { return MaterialType::Emitter; }

private:
    float3 m_color;
    float m_strength;
};

#else 
CALLABLE_FUNC void DC_FUNC(sample_emitter)(SurfaceInteraction* si, void* matdata) {
    const EmitterData* emitter = reinterpret_cast<EmitterData*>(matdata);
    // si->emission = emitter->color;
    si->radiance = make_float3(1.0f);
    si->trace_terminate = true;
}
#endif

}