#include "dielectric.h"

namespace prayground {

    // ------------------------------------------------------------------
    Dielectric::Dielectric(
        const SurfaceCallableID& surface_callable_id, 
        const std::shared_ptr<Texture>& texture, 
        float ior, 
        float absorb_coeff, 
        Sellmeier sellmeier, 
        Thinfilm thinfilm
    )
        : Material(surface_callable_id), 
        m_texture(texture), 
        m_ior(ior), 
        m_absorb_coeff(absorb_coeff), 
        m_sellmeier(sellmeier), 
        m_thinfilm(thinfilm)
    {

    }

    Dielectric::~Dielectric()
    {

    }

    // ------------------------------------------------------------------
    SurfaceType Dielectric::surfaceType() const 
    {
        return SurfaceType::Refraction;
    }

    // ------------------------------------------------------------------
    void Dielectric::copyToDevice()
    {
        Material::copyToDevice();
        
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();
        m_thinfilm.copyToDevice();
    
        Data data = this->getData();

        if (!d_data) 
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(Data),
            cudaMemcpyHostToDevice
        ));

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

    void Dielectric::free()
    {
        m_texture->free();
        Material::free();
    }

    // ------------------------------------------------------------------
    void Dielectric::setIor(const float ior)
    {
        m_ior = ior;
    }

    float Dielectric::ior() const 
    {
        return m_ior;
    }

    void Dielectric::setAbsorbCoeff(const float absorb_coeff)
    {
        m_absorb_coeff = absorb_coeff;
    }

    float Dielectric::absorbCoeff() const
    {
        return m_absorb_coeff;
    }

    void Dielectric::setSellmeierType(Sellmeier sellmeier)
    {
        m_sellmeier = sellmeier;
    }

    Sellmeier Dielectric::sellmeierType() const
    {
        return m_sellmeier;
    }

    void Dielectric::setThinfilm(const Thinfilm& thinfilm)
    {
        m_thinfilm = thinfilm;
    }
    Thinfilm Dielectric::thinfilm() const
    {
        return m_thinfilm;
    }


    // ------------------------------------------------------------------
    void Dielectric::setTexture(const std::shared_ptr<Texture>& texture)
    {
        m_texture = texture;
    }

    std::shared_ptr<Texture> Dielectric::texture() const 
    {
        return m_texture;
    }

    Dielectric::Data Dielectric::getData() const
    {
        return Data{
            .texture = m_texture->getData(),
            .ior = m_ior,
            .absorb_coeff = m_absorb_coeff,
            .sellmeier = m_sellmeier, 
            .thinfilm = m_thinfilm.getData()
        };
    }

} // namespace prayground