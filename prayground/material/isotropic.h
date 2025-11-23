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

    void copyToDevice() override
    {
        Data data = this->getData();

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(Data), cudaMemcpyHostToDevice));

        // Copy surface info to the device
        SurfaceInfo surface_info{
            .data = d_data,
            .callable_id = m_surface_callable_id,
            .type = surfaceType(),
            .use_bumpmap = useBumpmap(),
            .bumpmap = bumpmapData()
        };
        if (!d_surface_info)
            CUDA_CHECK(cudaMalloc(&d_surface_info, sizeof(SurfaceInfo)));
        CUDA_CHECK(cudaMemcpy(
            d_surface_info,
            &surface_info, sizeof(SurfaceInfo),
            cudaMemcpyHostToDevice
        ));
    }

    // Dummy override
    void setTexture(const std::shared_ptr<Texture>& texture) override {}
    std::shared_ptr<Texture> texture() const override { return nullptr; }

    Data getData() const { return { m_albedo }; }
private:
    Vec3f m_albedo;

#endif
};

}