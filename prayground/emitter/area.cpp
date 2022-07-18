#include "area.h"

namespace prayground {

    // ---------------------------------------------------------------------------
    AreaEmitter::AreaEmitter(const SurfaceCallableID& surface_callable_id, const std::shared_ptr<Texture>& texture, float intensity, bool twosided)
        : m_surface_callable_id(surface_callable_id), m_texture(texture), m_intensity(intensity), m_twosided(twosided)
    {

    }

    // ---------------------------------------------------------------------------
    SurfaceType AreaEmitter::surfaceType() const 
    {
        return SurfaceType::AreaEmitter;
    }

    SurfaceInfo AreaEmitter::surfaceInfo() const
    {
        ASSERT(d_data, "Area emitter data on device hasn't been allocated yet.");

        return SurfaceInfo{
            .data = d_data,
            .callable_id = m_surface_callable_id,
            .type = SurfaceType::AreaEmitter
        };
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

    const SurfaceCallableID& AreaEmitter::surfaceCallableID() const
    {
        return m_surface_callable_id;
    }

    void AreaEmitter::free()
    {
        
    }

    AreaEmitter::Data AreaEmitter::getData() const 
    {
        return { m_texture->getData(), m_intensity, m_twosided };
    }

} // namespace prayground