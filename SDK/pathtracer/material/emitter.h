#pragma once 

#include "../core/material.h"

namespace pt {

class Emitter final : public Material {
private:
    float3 color;
    float strength;

public:
    explicit HOSTDEVICE Emitter(float3 c, float s)
    : color(c), strength(s) {}

    HOSTDEVICE void sample(SurfaceInteraction& si) const override;
    HOSTDEVICE float3 emittance(SurfaceInteraction& si) const override;
    HOST MaterialType type() const override { return MaterialType::Emitter; }
};

HOSTDEVICE void Emitter::sample(SurfaceInteraction& si) const {
    si.trace_terminate = true;
}

HOSTDEVICE float3 Emitter::emittance(SurfaceInteraction& si) const {
    return color * strength;
}

}