#include "roughdielectric.h"

namespace prayground {
    RoughDielectric::RoughDielectric(const SurfaceCallableID& callable_id, float ior, float roughness, float absorb_coeff, Sellmeier sellmeier)
        : Material(callable_id), m_ior(ior), m_roughness(roughness), m_absorb_coeff(absorb_coeff), m_sellmeier(sellmeier)
    {
    }

    RoughDielectric::~RoughDielectric()
    {
    }

    SurfaceType RoughDielectric::surfaceType() const
    {
        return SurfaceType::RoughRefraction | SurfaceType::RoughReflection;
    }

    void RoughDielectric::copyToDevice()
    {
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();

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

    void RoughDielectric::free()
    {
        m_texture->free();
        Material::free();
    }

    void RoughDielectric::setTexture(const std::shared_ptr<Texture>& texture)
    {
        m_texture = texture;
    }

    std::shared_ptr<Texture> RoughDielectric::texture() const
    {
        return m_texture;
    }

    void RoughDielectric::setRoughness(const float roughness)
    {
        m_roughness = roughness;
    }

    const float& RoughDielectric::roughness() const
    {
        return m_roughness;
    }

    void RoughDielectric::setIor(const float ior)
    {
        m_ior = ior;
    }

    const float& RoughDielectric::ior() const
    {
        return m_ior;
    }

    void RoughDielectric::setAbsorbCoeff(const float absorb_coeff)
    {
        m_absorb_coeff = absorb_coeff;
    }
    const float& RoughDielectric::absorbCoeff() const
    {
        return m_absorb_coeff;
    }

    RoughDielectric::Data RoughDielectric::getData() const
    {
        return Data{
            .texture = m_texture->getData(), 
            .ior = m_ior, 
            .roughness = m_roughness,
            .absorb_coeff = m_absorb_coeff, 
            .sellmeier = m_sellmeier
        };
    }
} // namespace prayground