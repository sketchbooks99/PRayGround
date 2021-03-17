#pragma once 

#include <cuda/random.h>
#include <core/material.h>
#include <core/bsdf.h>

namespace pt {

class Dielectric final : public Material {
public:
    HOSTDEVICE Dielectric(float3 a, float ior) : albedo(a), ior(ior) {
        #ifndef __CUDACC__
        setup_on_device();
        #endif
    }

    HOSTDEVICE ~Dielectric() {
        #ifndef __CUDACC__
        delete_on_device();
        #endif
    }

    HOSTDEVICE void sample(SurfaceInteraction& si) const override;
    HOSTDEVICE float3 emittance(SurfaceInteraction& /* si */) const override { return make_float3(0.f); }

    HOST MaterialType type() const override { return MaterialType::Dielectric; }

private:
    HOST void setup_on_device() override {
        CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeof(MaterialPtr)));
        setup_object_on_device<<<1,1>>>((Dielectric**)&d_ptr, albedo, ior);
        CUDA_SYNC_CHECK();
    }

    HOST void delete_on_device() override {
        delete_object_on_device<<<1,1>>>(d_ptr);
        CUDA_SYNC_CHECK();
    }
    float3 albedo;
    float ior;
};

HOSTDEVICE void Dielectric::sample(SurfaceInteraction& si) const {
    si.attenuation = albedo;
    si.trace_terminate = false;
    
    float ni = 1.0f; // air
    float nt = ior;  // ior specified 
    float cosine = dot(si.wi, si.n);
    bool into = cosine < 0;
    float3 outward_normal = into ? si.n : -si.n;

    if (!into) swap(ni, nt);

    cosine = fabs(cosine);
    float sine = sqrtf(1.0 - cosine*cosine);
    bool cannot_refract = (ni / nt) * sine > 1.0f;

    float reflect_prob = fr(cosine, ni, nt);

    if (cannot_refract || reflect_prob > rnd(si.seed))
        si.wo = reflect(si.wi, outward_normal);
    else    
        si.wo = refract(si.wi, outward_normal, cosine, ni, nt);
}

}