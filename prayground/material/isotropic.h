#pragma once 

#include <prayground/core/material.h>

namespace prayground {

template <typename T>
class Isotropic_ final : public Material {
public:
    using ColorT = T;
    struct Data {
        T albedo;
    };

#ifndef __CUDACC__
    Isotropic_(const float3& albedo)
        : m_albedo(albedo) {}

    SurfaceType surfaceType() const override
    {
        return SurfaceType::Diffuse;
    }

    void copyToDevice() override
    {
        Data data{ .albedo = m_albedo };

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(Data), cudaMemcpyHostToDevice));
    }
private:
    Vec3f m_albedo;

#endif
};

}