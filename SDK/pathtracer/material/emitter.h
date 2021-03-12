#pragma once 

#include "../core/material.h"

namespace pt {

class Emitter final : public Material {
private:
    float3 color;
    float strength;

    HOST void setup_on_device() override {
        CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeof(MaterialPtr)));
        setup_material_on_device<<<1,1>>>((Emitter**)&d_ptr, color, strength);
        CUDA_SYNC_CHECK();
    }

    HOST void delete_on_device() override {
        delete_material_on_device<<<1,1>>>(d_ptr);
        CUDA_SYNC_CHECK();
    }

public:
    explicit HOSTDEVICE Emitter(float3 c, float s)
    : color(c), strength(s) {
        #ifndef __CUDACC__
        setup_on_device();
        #endif
    }

    HOSTDEVICE ~Emitter() {
        #ifndef __CUDACC__
        delete_on_device();
        #endif
    }

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