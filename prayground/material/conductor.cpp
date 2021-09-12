#include "conductor.h"

namespace prayground {

// ------------------------------------------------------------------
Conductor::Conductor(const std::shared_ptr<Texture>& texture, bool twosided)
: m_texture(texture), m_twosided(twosided)
{
    
}

Conductor::~Conductor()
{

}

// ------------------------------------------------------------------
SurfaceType Conductor::surfaceType() const
{
    return SurfaceType::Reflection;
}

// ------------------------------------------------------------------
void Conductor::copyToDevice()
{
    if (!m_texture->devicePtr())
        m_texture->copyToDevice();

    ConductorData data = {
        .tex_data = m_texture->devicePtr(), 
        .twosided = m_twosided,
        .tex_program_id = m_texture->programId()
    };

    if (!d_data)
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(ConductorData)));
    CUDA_CHECK(cudaMemcpy(
        d_data,
        &data, sizeof(ConductorData), 
        cudaMemcpyHostToDevice
    ));
}

void Conductor::free()
{
    m_texture->free();
    Material::free();
}

// ------------------------------------------------------------------
void Conductor::setTexture(const std::shared_ptr<Texture>& texture)
{
    m_texture = texture;
}

std::shared_ptr<Texture> Conductor::texture() const 
{
    return m_texture;
}

} // ::prayground