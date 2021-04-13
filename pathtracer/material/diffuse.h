#pragma once

#include <cuda/random.h>
#include "../core/material.h"
#include "../core/bsdf.h"
#include "../core/onb.h"
#include "../optix/sbt.h"
#include "../texture/constant.h"

namespace pt {

struct DiffuseData {
    // float3 albedo;
    void* texdata;
    unsigned int tex_func_idx;
};

#ifndef __CUDACC__

class Diffuse final : public Material {
public:
    explicit Diffuse(float3 a) : m_texture(new ConstantTexture(a)) { }
    explicit Diffuse(Texture* texture) : m_texture(texture) {}

    ~Diffuse() { }

    void sample(SurfaceInteraction& si) const override {
        unsigned int seed = si.seed;
        si.trace_terminate = false;

        {
            const float z1 = rnd(seed);
            const float z2 = rnd(seed);

            float3 w_in = cosine_sample_hemisphere(z1, z2);
            Onb onb(si.n);
            onb.inverse_transform(w_in);
            si.wo = w_in;
        }

        si.seed = seed;
        si.attenuation = m_texture->eval(si);
    }
    
    float3 emittance(SurfaceInteraction& /* si */) const override { return make_float3(0.f); }

    void prepare_data() override {
        m_texture->prepare_data();

        DiffuseData data {
            reinterpret_cast<void*>(m_texture->get_dptr()),
            static_cast<unsigned int>(m_texture->type()) + static_cast<unsigned int>(MaterialType::Count)
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
    // float3 m_albedo;
    Texture* m_texture;
};

#else
/**
 * \brief Sample bsdf at the surface.
 * \note This is direct callables function on the device, 
 *       so this function cannot launch `optixTrace()`.  
 */
CALLABLE_FUNC void CC_FUNC(sample_diffuse)(SurfaceInteraction* si, void* matdata) {
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(matdata);

    unsigned int seed = si->seed;
    si->trace_terminate = false;

    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in = cosine_sample_hemisphere(z1, z2);
        Onb onb(si->n);
        onb.inverse_transform(w_in);
        si->wo = w_in;
    }
    si->seed = seed;
    // si->attenuation = diffuse->albedo;
    // si->attenuation = make_float3(0.8f);
    si->attenuation = optixDirectCall<float3, SurfaceInteraction*, void*>(diffuse->tex_func_idx, si, diffuse->texdata);
    si->emission = make_float3(0.0f);
}

#endif

}