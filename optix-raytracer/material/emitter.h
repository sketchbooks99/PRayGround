#pragma once 

#include "../core/material.h"
#include "../texture/constant.h"

namespace oprt {

struct EmitterData {
    void* texdata;
    float strength;
    bool twosided;
    unsigned int tex_func_idx;
};

#ifndef __CUDACC__
class Emitter final : public Material {
public:
    Emitter(const float3& color, float strength=1.0f, bool twosided=true) 
    : m_texture(new ConstantTexture(color)), m_strength(strength), m_twosided(twosided) { }
    Emitter(Texture* texture, float strength=1.0f, bool twosided=true)
    : m_texture(texture), m_strength(strength), m_twosided(twosided) {}

    ~Emitter() { }

    void prepare_data() override {
        m_texture->prepare_data();

        EmitterData data = {
            m_texture->get_dptr(), 
            m_strength,
            m_twosided,
            static_cast<unsigned int>(m_texture->type()) + static_cast<unsigned int>(MaterialType::Count)
        };

        CUDA_CHECK(cudaMalloc(&d_data, sizeof(EmitterData)));
        CUDA_CHECK(cudaMemcpy(
            d_data, 
            &data, sizeof(EmitterData), 
            cudaMemcpyHostToDevice
        ));
    }

    MaterialType type() const override { return MaterialType::Emitter; }

private:
    Texture* m_texture;
    float m_strength;
    bool m_twosided;
};

#else 

CALLABLE_FUNC void CC_FUNC(sample_emitter)(SurfaceInteraction* si, void* matdata) {
    const EmitterData* emitter = reinterpret_cast<EmitterData*>(matdata);

    float is_emitted = 1.0f;
    if (!emitter->twosided)
        is_emitted = dot(si->wi, si->n) < 0.0f ? 1.0f : 0.0f;

    si->emission = optixDirectCall<float3, SurfaceInteraction*, void*>(
        emitter->tex_func_idx, si, emitter->texdata) * emitter->strength;
    si->trace_terminate = true;
}

#endif

}