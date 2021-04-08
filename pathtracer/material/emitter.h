#pragma once 

#include <include/core/material.h>

namespace pt {

struct EmitterData {
    float3 color;
    float strength;
};

class Emitter final : public Material {
public:
    Emitter(const float3& color, float strength) 
    : m_color(color), m_strength(strength) { }

    ~Emitter() { }

    void sample(SurfaceInteraction& si) const override {
        si.trace_terminate = true;
    }
    float3 emittance(SurfaceInteraction& si) const override {
        return m_color * m_strength;
    }

    float3 emitted() const { return m_color; }

    size_t member_size() const override { return sizeof(m_color) + sizeof(m_strength); }

    MaterialType type() const override { return MaterialType::Emitter; }

private:

    float3 m_color;
    float m_strength;
};

}