#include "dielectric.h"

namespace prayground {

// ------------------------------------------------------------------
Dielectric::Dielectric(const std::shared_ptr<Texture>& texture, float ior)
: m_texture(texture), m_ior(ior)
{

}

Dielectric::~Dielectric()
{

}

// ------------------------------------------------------------------
void Dielectric::copyToDevice()
{
    if (!m_texture->devicePtr())
        m_texture->copyToDevice();

    DielectricData data = {
        .tex_data = m_texture->devicePtr(), 
        .ior = m_ior, 
        .tex_program_id = m_texture->programId()
    };

    if (!d_data) 
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(DielectricData)));
    CUDA_CHECK(cudaMemcpy(
        d_data,
        &data, sizeof(DielectricData),
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