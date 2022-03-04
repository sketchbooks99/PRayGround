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
SurfaceType Diffuse::surfaceType() const 
{
    return SurfaceType::Diffuse;
}

// ------------------------------------------------------------------
void Diffuse::copyToDevice()
{
    if (!m_texture->devicePtr())
        m_texture->copyToDevice();

    Data data {
        .texture = m_texture->getData(),
        .twosided = m_twosided
    };

    if (!d_data)
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
    CUDA_CHECK(cudaMemcpy(
        d_data,
        &data, sizeof(Data), 
        cudaMemcpyHostToDevice
    ));
}

void Diffuse::free()
{
    m_texture->free();
    Material::free();
}

}