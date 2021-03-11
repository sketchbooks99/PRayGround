#pragma once

#include "../core/material.h"
#include "../core/bsdf.h"

namespace pt {

class Conductor final : public Material {
private:
    float3 albedo;
    float fuzz;

public:
    explicit HOSTDEVICE Conductor(float3 a, float f) : albedo(a), fuzz(f) {}

    HOSTDEVICE void sample(SurfaceInteraction& si) const override;
    HOSTDEVICE float3 emittance(SurfaceInteraction& si) const override { return make_float3(0.f); }

    HOST MaterialType type() const override { return MaterialType::Conductor; }
};

HOSTDEVICE void Conductor::sample(SurfaceInteraction& si) const {
    si.wo = reflect(si.wi, si.n);
    si.attenuation = albedo;
    si.trace_terminate = false;
}

}



