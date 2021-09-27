#include "isotropic.h"

namespace prayground {

Isotropic::Isotropic(const float3& albedo) : m_albedo(albedo)
{

}

// ------------------------------------------------------------------
SurfaceType Isotropic::surfaceType() const 
{
    return SurfaceType::Diffuse;
}

// ------------------------------------------------------------------
void Isotropic::copyToDevice()
{
    IsotropicData data {
        .albedo = m_albedo
    };

    if (!d_data)
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(IsotropicData)));
    CUDA_CHECK(cudaMemcpy(
        d_data, &data, sizeof(IsotropicData), cudaMemcpyHostToDevice
    ));
}

} // ::prayground