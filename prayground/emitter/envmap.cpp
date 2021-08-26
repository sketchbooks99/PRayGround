#include "envmap.h"

namespace prayground {

void EnvironmentEmitter::copyToDevice()
{
    if (!m_texture->devicePtr())
        m_texture->copyToDevice();

    EnvironmentEmitterData data =
    {
        m_texture->devicePtr(),
        m_texture->programId()
    };

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(EnvironmentEmitterData)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_data), 
        &data, sizeof(EnvironmentEmitterData), 
        cudaMemcpyHostToDevice
    ));
}

} // ::prayground
