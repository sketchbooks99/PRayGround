#pragma once

#include <cuda/random.h>
#include <include/core/material.h>
#include <include/core/bsdf.h>
#include <include/core/onb.h>

namespace pt {

class Diffuse final : public Material {
public:
    explicit HOSTDEVICE Diffuse(float3 a)
    : m_albedo(a)
    {
#ifndef __CUDACC__
        setup_on_device();
#endif
    }
    HOSTDEVICE ~Diffuse() {
#ifndef __CUDACC__
        delete_on_device();
#endif
    }

    HOSTDEVICE void sample(SurfaceInteraction& si) const override {
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

            si.attenuation *= m_albedo;
        }

        const float z1 = rnd(seed);
        const float z2 = rnd(seed);
        si.seed = seed;
        si.radiance = m_albedo;

        // ParallelgramLight light = param.light;
        // // Sample emitter position
        // const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

        // // Calculate properties of light sample (for area based pdf)
        // const float Ldist = length(light_pos - si.p);
        // const float3 L = normalize(light_pos - si.p);
        // const float nDl = dot(si.n, L);
        // const float LnDl = -dot(light.normal, L);
    }
    
    HOSTDEVICE float3 emittance(SurfaceInteraction& /* si */) const override { return make_float3(0.f); }

    HOSTDEVICE float3 albedo() const { return m_albedo; }

#ifndef __CUDACC__
    MaterialType type() const override { return MaterialType::Diffuse; }
#endif

private:
    void setup_on_device() override;
    void delete_on_device() override;

    float3 m_albedo;
};

}