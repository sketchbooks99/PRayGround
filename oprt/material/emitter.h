#pragma once 

#include "../core/material.h"
#include "../texture/constant.h"

namespace oprt {

struct EmitterData {
    void* texdata;
    float strength;
    bool twosided;
    unsigned int tex_func_id;
};

#ifndef __CUDACC__
class Emitter final : public Material {
public:
    Emitter(const float3& color, float strength=1.0f, bool twosided=true) 
    : m_texture(new ConstantTexture(color)), m_strength(strength), m_twosided(twosided) { }
    Emitter(Texture* texture, float strength=1.0f, bool twosided=true)
    : m_texture(texture), m_strength(strength), m_twosided(twosided) {}

    ~Emitter() { }

    void prepareData() override {
        m_texture->prepareData();

        EmitterData data = {
            m_texture->devicePtr(), 
            m_strength,
            m_twosided,
            static_cast<unsigned int>(m_texture->type()) + static_cast<unsigned int>(MaterialType::Count) * 2
        };

        CUDA_CHECK(cudaMalloc(&d_data, sizeof(EmitterData)));
        CUDA_CHECK(cudaMemcpy(
            d_data, 
            &data, sizeof(EmitterData), 
            cudaMemcpyHostToDevice
        ));
    }

    void freeData() override
    {
        m_texture->freeData();
    }

    MaterialType type() const override { return MaterialType::Emitter; }

private:
    std::shared_ptr<Texture> m_texture;
    float m_strength;
    bool m_twosided;
};

#else 

CALLABLE_FUNC void DC_FUNC(sample_emitter)(SurfaceInteraction* si, void* matdata) {
    const EmitterData* emitter = reinterpret_cast<EmitterData*>(matdata);
    si->trace_terminate = true;
}

CALLABLE_FUNC float3 CC_FUNC(bsdf_emitter)(SurfaceInteraction* si, void* matdata)
{
    const EmitterData* emitter = reinterpret_cast<EmitterData*>(matdata);

    float is_emitted = 1.0f;
    if (!emitter->twosided)
        is_emitted = dot(si->wi, si->n) < 0.0f ? 1.0f : 0.0f;

    si->emission = optixDirectCall<float3, SurfaceInteraction*, void*>(
        emitter->tex_func_id, si, emitter->texdata) * emitter->strength * is_emitted;
    return make_float3(0.0f);
}

CALLABLE_FUNC float DC_FUNC(pdf_emitter)(SurfaceInteraction* si, void* matdata)
{
    return 1.0f;
}

#endif

}