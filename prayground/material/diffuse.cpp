#include "diffuse.h"

namespace prayground {

// ------------------------------------------------------------------
Diffuse::Diffuse(const std::shared_ptr<Texture>& texture, bool twosided)
: m_texture(texture), m_twosided(twosided)
{

}

Diffuse::~Diffuse()
{

}

// ------------------------------------------------------------------
void Diffuse::copyToDevice()
{
    if (!m_texture->devicePtr())
        m_texture->copyToDevice();

    DiffuseData data {
        .tex_data = m_texture->devicePtr(),
        .twosided = m_twosided,
        .tex_program_id = m_texture->programId()
    };

    if (!d_data)
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(DiffuseData)));
    CUDA_CHECK(cudaMemcpy(
        d_data,
        &data, sizeof(DiffuseData), 
        cudaMemcpyHostToDevice
    ));
}

void Diffuse::free()
{
    m_texture->free();
    Material::free();
}

}