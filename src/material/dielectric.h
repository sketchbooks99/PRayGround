#pragma once 

#include <cuda/random.h>
#include <include/core/material.h>
#include <include/core/bsdf.h>

namespace pt {

class Dielectric final : public Material {
public:
    HOSTDEVICE Dielectric(float3 a, float ior);
    HOSTDEVICE ~Dielectric();

    HOSTDEVICE void sample(SurfaceInteraction& si) const override {
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
    
    HOSTDEVICE float3 emittance(SurfaceInteraction& /* si */) const override { return make_float3(0.f); }
    HOST MaterialType type() const override { return MaterialType::Dielectric; }
private:
    HOST void setup_on_device() override;
    HOST void delete_on_device() override;

    float3 m_albedo;
    float m_ior;
};

}