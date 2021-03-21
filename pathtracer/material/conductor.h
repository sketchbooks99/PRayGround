#pragma once

#include <include/core/material.h>
#include <include/core/bsdf.h>

namespace pt {

class Conductor;

class Conductor final : public Material {
public:
    HOSTDEVICE Conductor(float3 a, float f);
    HOSTDEVICE ~Conductor();

    HOSTDEVICE void sample(SurfaceInteraction& si) const override {
        si.wo = reflect(si.wi, si.n);
        si.attenuation = m_albedo;
        si.trace_terminate = false;
    }
    
    HOSTDEVICE float3 emittance(SurfaceInteraction& si) const override { return make_float3(0.f); }

#ifndef __CUDACC__
    MaterialType type() const override { return MaterialType::Conductor; }
#endif

private:
    void setup_on_device() override;
    void delete_on_device() override;

    float3 m_albedo;
    float m_fuzz;
};

}



