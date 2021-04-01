#pragma once 

#include <include/core/material.h>

namespace pt {

class Emitter final : public Material {
public:
    explicit HOSTDEVICE Emitter(float3 color, float strength);

    HOSTDEVICE ~Emitter();

    HOSTDEVICE void sample(SurfaceInteraction& si) const override {
        si.trace_terminate = true;
    }
    HOSTDEVICE float3 emittance(SurfaceInteraction& si) const override {
        return m_color * m_strength;
    }

    MaterialType type() const override { return MaterialType::Emitter; }
    float3 emitted() const { return m_color; }

private:
    HOST void setup_on_device() override;
    HOST void delete_on_device() override;

    float3 m_color;
    float m_strength;
};

}