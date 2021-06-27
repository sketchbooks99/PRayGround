#include "area.h"
#include "../core/material.h"

namespace oprt {
    
void AreaEmitter::prepareData() 
{
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

}