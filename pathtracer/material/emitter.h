#pragma once 

#include <include/core/material.h>

namespace pt {

class Emitter final : public Material {
public:
    explicit HOSTDEVICE Emitter(const float3& color, float strength) 
    : m_color(color), m_strength(strength)
    {
#ifndef __CUDACC__
        setup_on_device();
#endif
    }

    HOSTDEVICE ~Emitter() {
#ifndef __CUDACC__
        delete_on_device();
#endif
    }

    HOSTDEVICE void sample(SurfaceInteraction& si) const override {
        si.trace_terminate = true;
    }
    HOSTDEVICE float3 emittance(SurfaceInteraction& si) const override {
        return m_color * m_strength;
    }

    HOSTDEVICE float3 emitted() const { return m_color; }

    HOSTDEVICE size_t member_size() const override { return sizeof(m_color) + sizeof(m_strength); }

#ifndef __CUDACC__
    HOST MaterialType type() const override { return MaterialType::Emitter; }
#endif

private:
    void setup_on_device() override;
    void delete_on_device() override;

    float3 m_color;
    float m_strength;
};

}