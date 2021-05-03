#pragma once 

#include <cuda/random.h>
#include "../core/material.h"
#include "../core/bsdf.h"
#include "../texture/constant.h"

namespace oprt {

struct DielectricData {
    void* texdata;
    float ior;
    unsigned int tex_func_idx;
};

#ifndef __CUDACC__

class Dielectric final : public Material {
public:
    Dielectric(const float3& a, float ior)
    : m_texture(new ConstantTexture(a)), m_ior(ior) { }
    Dielectric(Texture* texture, float ior)
    : m_texture(texture), m_ior(ior) {}
    ~Dielectric() { }

    void prepare_data() override {
        m_texture->prepare_data();

        DielectricData data = {
            m_texture->get_dptr(), 
            m_ior, 
            static_cast<unsigned int>(m_texture->type()) + static_cast<unsigned int>(MaterialType::Count) 
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
    // float3 m_albedo;
    Texture* m_texture;
    float m_ior;
};

#else 
CALLABLE_FUNC void CC_FUNC(sample_dielectric)(SurfaceInteraction* si, void* matdata) {
    const DielectricData* dielectric = reinterpret_cast<DielectricData*>(matdata);

    si->attenuation *= optixDirectCall<float3, SurfaceInteraction*, void*>(dielectric->tex_func_idx, si, dielectric->texdata);
    si->trace_terminate = false;
    si->radiance = make_float3(0.0f);

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
    unsigned int seed = si->seed;

    if (cannot_refract || reflect_prob > rnd(seed))
        si->wo = reflect(si->wi, outward_normal);
    else    
        si->wo = refract(si->wi, outward_normal, cosine, ni, nt);
    si->emission = make_float3(0.0f);
    si->seed = seed;
    si->radiance_evaled = false;
}

#endif

}