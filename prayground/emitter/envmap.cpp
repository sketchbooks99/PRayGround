#include "envmap.h"

namespace prayground {

    void EnvironmentEmitter::copyToDevice()
    {
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();

        auto data = this->getData();

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_data), 
            &data, sizeof(Data), 
            cudaMemcpyHostToDevice
        ));
    }

    EnvironmentEmitter::Data EnvironmentEmitter::getData() const 
    {
        return { m_texture->getData() };
    }

} // namespace prayground
