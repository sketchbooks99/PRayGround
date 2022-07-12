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
    Isotropic_(const SurfaceCallableID& surface_callable_id, const float3& albedo)
        : Material(surface_callable_id), m_albedo(albedo) {}

    SurfaceType surfaceType() const override
    {
        return SurfaceType::Diffuse;
    }

    SurfaceInfo surfaceInfo() const override
    {
        ASSERT(d_data, "Material data on device hasn't been allocated yet.");

        return SurfaceInfo{
            .data = d_data,
            .callable_id = m_surface_callable_id,
            .type = SurfaceType::Diffuse
        };
    }

    void copyToDevice() override
    {
        Data data = this->getData();

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(Data), cudaMemcpyHostToDevice));
    }

    Data getData() const { return { m_albedo }; }
private:
    Vec3f m_albedo;

#endif
};

}