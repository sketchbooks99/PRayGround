#include "area.h"
#include "../core/material.h"

namespace oprt {

// ---------------------------------------------------------------------------
AreaEmitter::AreaEmitter(const float3& color, float intensity, bool twosided)
: (std::make_shared<ConstantTexture>(color)), m_intensity(intensity), m_twosided(twosided)
{

}

AreaEmitter::AreaEmitter(const std::shared_ptr<Texture>& texture, float intensity=1.0f, bool twosided)
: m_texture(texture), m_intensity(intensity), m_twosided(twosided)
{

}
    
// ---------------------------------------------------------------------------
void AreaEmitter::prepareData() 
{
    if (!m_texture->devicePtr())
        m_texture->prepareData();

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

void AreaEmitter::freeData()
{

}

}