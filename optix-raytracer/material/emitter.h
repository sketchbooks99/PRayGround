#pragma once 

#include "../core/material.h"
#include "../texture/constant.h"

namespace oprt {

struct EmitterData {
    void* texdata;
    float strength;
    unsigned int tex_func_idx;
};

#ifndef __CUDACC__
class Emitter final : public Material {
public:
    Emitter(const float3& color, float strength = 1.0f) 
    : m_texture(new ConstantTexture(color)), m_strength(strength) { }
    Emitter(Texture* texture, float strength = 1.0f)
    : m_texture(texture), m_strength(strength) {}

    ~Emitter() { }

    void prepare_data() override {
        m_texture->prepare_data();

        EmitterData data = {
            reinterpret_cast<void*>(m_texture->get_dptr()), 
            m_strength,
            static_cast<unsigned int>(m_texture->type()) + static_cast<unsigned int>(MaterialType::Count)
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
    Texture* m_texture;
    float m_strength;
};

#else 
CALLABLE_FUNC void CC_FUNC(sample_emitter)(SurfaceInteraction* si, void* matdata) {
    const EmitterData* emitter = reinterpret_cast<EmitterData*>(matdata);
    si->emission = optixDirectCall<float3, SurfaceInteraction*, void*>(
        emitter->tex_func_idx, si, emitter->texdata) * emitter->strength;
    si->attenuation = make_float3(0.0f);
    si->trace_terminate = true;
}
#endif

}