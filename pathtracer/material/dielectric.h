#pragma once 

#include <cuda/random.h>
#include <include/core/material.h>
#include <include/core/bsdf.h>

namespace pt {

struct DielectricData {
    float3 albedo;
    float ior;
};

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
    }
    
    float3 emittance( SurfaceInteraction& /* si */ ) const override { return make_float3(0.f); }

    size_t member_size() const override { return sizeof(m_albedo) + sizeof(m_ior); }

    MaterialType type() const override { return MaterialType::Dielectric; }

private:
    float3 m_albedo;
    float m_ior;
};

}