#include "dielectric.h"

namespace prayground {

// ------------------------------------------------------------------
Dielectric::Dielectric(const std::shared_ptr<Texture>& texture, float ior, float absorb_coeff, Sellmeier sellmeier)
: m_texture(texture), m_ior(ior), m_absorb_coeff(absorb_coeff), m_sellmeier(sellmeier)
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
    if (!m_texture->devicePtr())
        m_texture->copyToDevice();
    
    Data data = {
        .texture = m_texture->getData(),
        .ior = m_ior, 
        .absorb_coeff = m_absorb_coeff,
        .sellmeier = m_sellmeier
    };

    if (!d_data) 
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
    CUDA_CHECK(cudaMemcpy(
        d_data,
        &data, sizeof(Data),
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

// ------------------------------------------------------------------
void Dielectric::setTexture(const std::shared_ptr<Texture>& texture)
{
    m_texture = texture;
}

std::shared_ptr<Texture> Dielectric::texture() const 
{
    return m_texture;
}

} // ::prayground