#include "area.h"

namespace prayground {

    // ---------------------------------------------------------------------------
    AreaEmitter::AreaEmitter(const std::shared_ptr<Texture>& texture, float intensity, bool twosided)
    : m_texture(texture), m_intensity(intensity), m_twosided(twosided)
    {

    }

    // ---------------------------------------------------------------------------
    SurfaceType AreaEmitter::surfaceType() const 
    {
        return SurfaceType::AreaEmitter;
    }
        
    // ---------------------------------------------------------------------------
    void AreaEmitter::copyToDevice() 
    {
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();

        auto data = this->getData();

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(Data),
            cudaMemcpyHostToDevice
        ));
    }

    void AreaEmitter::setTexture(const std::shared_ptr<Texture>& texture)
    {
        m_texture = texture;
    }

    std::shared_ptr<Texture> AreaEmitter::texture() const
    {
        return m_texture;
    }

    void AreaEmitter::setIntensity(float intensity)
    {
        m_intensity = intensity;
    }

    float AreaEmitter::intensity() const 
    {
        return m_intensity;
    }

    void AreaEmitter::free()
    {
        
    }

    AreaEmitter::Data AreaEmitter::getData() const 
    {
        return { m_texture->getData(), m_intensity, m_twosided };
    }

}