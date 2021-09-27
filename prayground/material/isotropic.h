#pragma once 

#include <prayground/core/material.h>

namespace prayground {

struct IsotropicData
{
    float3 albedo;
};

#ifndef __CUDACC__
class Isotropic final : public Material {
public:
    Isotropic(const float3& albedo);

    SurfaceType surfaceType() const override;

    void copyToDevice() override;
private:
    float3 m_albedo;
};
#endif

}