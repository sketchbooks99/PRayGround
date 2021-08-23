#include "area.h"
#include <oprt/core/material.h>

namespace oprt {

// ---------------------------------------------------------------------------
AreaEmitter::AreaEmitter(const std::shared_ptr<Texture>& texture, float intensity, bool twosided)
: m_texture(texture), m_intensity(intensity), m_twosided(twosided)
{

}
    
// ---------------------------------------------------------------------------
void AreaEmitter::copyToDevice() 
{
    if (!m_texture->devicePtr())
        m_texture->copyToDevice();

    AreaEmitterData data = {
        m_texture->devicePtr(),
        m_intensity, 
        m_twosided, 
        static_cast<unsigned int>(m_texture->type()) + static_cast<unsigned int>(MaterialType::Count) * 2
    };

    CUDA_CHECK(cudaMalloc(&d_data, sizeof(AreaEmitterData)));
    CUDA_CHECK(cudaMemcpy(
        d_data,
        &data, sizeof(AreaEmitterData),
        cudaMemcpyHostToDevice
    ));
}

void AreaEmitter::setProgramId(const int32_t prg_id)
{
    m_prg_id = prg_id;
}

int32_t AreaEmitter::programId() const 
{
    return m_prg_id;
}

std::shared_ptr<Texture> AreaEmitter::texture() const
{
    return m_texture;
}

void AreaEmitter::free()
{

}

}