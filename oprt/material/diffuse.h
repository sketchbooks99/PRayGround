#pragma once

#include <cuda/random.h>
#include "../core/material.h"
#include "../core/bsdf.h"
#include "../core/onb.h"
#include "../optix/sbt.h"
#include "../texture/constant.h"

namespace oprt {

struct DiffuseData {
    void* texdata;
    bool twosided;
    unsigned int tex_func_id;
};

#ifndef __CUDACC__

class Diffuse final : public Material {
public:
    explicit Diffuse(const float3& a, bool twosided=true)
    : m_texture(new ConstantTexture(a)), m_twosided(twosided) { }

    explicit Diffuse(Texture* texture, bool twosided=true)
    : m_texture(texture), m_twosided(twosided) {}

    explicit Diffuse(const std::shared_ptr<Texture>& texture, bool twosided=true)
    : m_texture(texture), m_twosided(twosided) {}

    ~Diffuse() { }

    void prepareData() override {
        m_texture->prepareData();

        DiffuseData data {
            m_texture->devicePtr(),
            m_twosided,
            static_cast<unsigned int>(m_texture->type()) + static_cast<unsigned int>(MaterialType::Count) * 2
        };

        CUDA_CHECK(cudaMalloc(&d_data, sizeof(DiffuseData)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(DiffuseData), 
            cudaMemcpyHostToDevice
        ));
    }

    void freeData() override
    {
        m_texture->freeData();
    }

    MaterialType type() const override { return MaterialType::Diffuse; }

private:
    std::shared_ptr<Texture> m_texture;
    bool m_twosided;
};

#else
/**
 * \brief Sample bsdf at the surface.
 * \note This is direct callables function on the device, 
 *       so this function cannot launch `optixTrace()`.  
 */

CALLABLE_FUNC void DC_FUNC(sample_diffuse)(SurfaceInteraction* si, void* matdata) {
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(matdata);

    if (diffuse->twosided)
        si->n = faceforward(si->n, -si->wi, si->n);

    unsigned int seed = si->seed;
    si->trace_terminate = false;
    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in = cosineSampleHemisphere(z1, z2);
        Onb onb(si->n);
        onb.inverseTransform(w_in);
        si->wo = w_in;
    }
    si->seed = seed;
}

CALLABLE_FUNC float3 CC_FUNC(bsdf_diffuse)(SurfaceInteraction* si, void* matdata)
{
    const DiffuseData* diffuse = reinterpret_cast<DiffuseData*>(matdata);
    const float3 albedo = optixDirectCall<float3, SurfaceInteraction*, void*>(diffuse->tex_func_id, si, diffuse->texdata);
    const float cosine = fmaxf(0.0f, dot(si->n, si->wo));
    si->emission = make_float3(0.0f);
    si->radiance_evaled = false;

    // Next event estimation
    float3 light_emission = make_float3(0.8f, 0.8f, 0.7f) * 15.0f;
    unsigned int seed = si->seed;
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    si->seed = seed;

    float3 v1 = make_float3(-130.0f, 0.0f, 0.0f);
    float3 v2 = make_float3(0.0f, 0.0f, 105.0f);
    const float3 light_pos = make_float3(343.0f, 548.6f, 227.0f) + v1*z1 + v2*z2;
    
    const float Ldist = length(light_pos - si->p);
    const float3 L = normalize(light_pos - si->p);
    const float nDl = dot(si->n, L);
    const float LnDl = -dot(make_float3(0.0f, -1.0f, 0.0f), L);
    float weight = 0.0f;
    if (nDl > 0.0f && LnDl > 0.0f)
    {
        const bool occluded = traceOcclusion(
            params.handle, 
            si->p, 
            L, 
            0.01f, 
            Ldist - 0.01f
        );

        if (!occluded)
        {
            const float A = length(cross(v1, v2));
            weight = nDl * LnDl * A / (M_PIf * Ldist * Ldist);
        }
    }
    si->radiance_evaled = true;
    si->emission = light_emission * make_float3(weight);
    return albedo * (cosine / M_PIf);
}

CALLABLE_FUNC float DC_FUNC(pdf_diffuse)(SurfaceInteraction* si, void* matdata)
{
    const float cosine = fmaxf(0.0f, dot(si->n, si->wo));
    return cosine / M_PIf;
}

#endif

}