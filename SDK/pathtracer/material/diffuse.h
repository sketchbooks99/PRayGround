#pragma once

#include <cuda/random.h>
#include "../core/material.h"
#include "../core/bsdf.h"

namespace pt {

class Diffuse final : public Material {

private:
    float3 albedo;

public:
    explicit HOSTDEVICE Diffuse(float3 a) : albedo(a) {}

    HOSTDEVICE void sample(SurfaceInteraction& si) const override;
    HOSTDEVICE float3 emittance(SurfaceInteraction& /* si */) const override { return make_float3(0.f); }

    HOSTDEVICE MaterialType type() const override { return MaterialType::Diffuse; }
};

HOSTDEVICE void sample(SurfaceInteraction& si) {
    si.attenuation = si;
    unsigned int seed = si.seed;

    {
        const float z1 = _rnd(seed);
        const float z2 = _rnd(seed);

        float3 w_in; 
        cosine_sample_hemisphere(z1, z2, w_in);
        Onb onb(si.n);
        onb.inverse_transform(w_in);
        si.wo = w_in;

        si.attenuation *= albedo;
    }

    const float z1 = _rnd(seed);
    const float z2 = _rnd(seed);
    prd.seed = seed;

    ParallelgramLight light = param.light;
    // Sample emitter position
    const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // Calculate properties of light sample (for area based pdf)
    const float Ldist = length(light_pos - si.p);
    const float3 L = normalize(light_pos - si.p);
    const float nDl = dot(si.n, L);
    const float LnDl = -dot(light.normal, L);

    float weight = 0.0f;
    if (nDl > 0.0f && LnDl > 0.0f)
    {
    }
}

}