#pragma once

#include <cuda/random.h>
#include <core/material.h>
#include <core/bsdf.h>
#include <core/onb.h>

namespace pt {

class Diffuse final : public Material {
public:
    explicit HOSTDEVICE Diffuse(float3 a) : albedo(a) {
        #ifndef __CUDACC__
        setup_on_device();
        #endif
    }

    HOSTDEVICE ~Diffuse() {
        #ifndef __CUDACC__
        delete_on_device();
        #endif
    }

    HOSTDEVICE void sample(SurfaceInteraction& si) const override;
    HOSTDEVICE float3 emittance(SurfaceInteraction& /* si */) const override { return make_float3(0.f); }

    HOSTDEVICE MaterialType type() const override { return MaterialType::Diffuse; }
private:
    HOST void setup_on_device() override {
        CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeof(MaterialPtr)));
        setup_material_on_device<<<1,1>>>((Diffuse**)&d_ptr, albedo);
        CUDA_SYNC_CHECK();
    }

    HOST void delete_on_device() override {
        delete_material_on_device<<<1,1>>>(d_ptr);
        CUDA_SYNC_CHECK();
    }

    float3 albedo;
};

HOSTDEVICE void Diffuse::sample(SurfaceInteraction& si) const {
    unsigned int seed = si.seed;
    si.trace_terminate = false;

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
    si.seed = seed;

    // ParallelgramLight light = param.light;
    // // Sample emitter position
    // const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

    // // Calculate properties of light sample (for area based pdf)
    // const float Ldist = length(light_pos - si.p);
    // const float3 L = normalize(light_pos - si.p);
    // const float nDl = dot(si.n, L);
    // const float LnDl = -dot(light.normal, L);
}

}