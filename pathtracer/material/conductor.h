#pragma once

#include <include/core/material.h>
#include <include/core/bsdf.h>

namespace pt {

struct ConductorData {
    float3 albedo; 
    float fuzz;
};

class Conductor final : public Material {
public:
    Conductor(float3 a, float f) 
    : m_albedo(a), m_fuzz(f) {}
    ~Conductor() {}

    void sample( SurfaceInteraction& si ) const override {
        si.wo = reflect(si.wi, si.n);
        si.attenuation = m_albedo;
        si.trace_terminate = false;
    }
    
    float3 emittance( SurfaceInteraction& si ) const override { return make_float3(0.f); }

    size_t member_size() const override { return sizeof(m_albedo) + sizeof(m_fuzz); } 

    MaterialType type() const override { return MaterialType::Conductor; }
    
private:
    float3 m_albedo;
    float m_fuzz;
};

}



