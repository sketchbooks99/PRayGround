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

        // Copy surface info to the device
        SurfaceInfo surface_info{
            .data = d_data,
            .callable_id = m_surface_callable_id,
            .type = surfaceType(),
            .use_bumpmap = false,
            .bumpmap = nullptr
        };
        if (!d_surface_info)
            CUDA_CHECK(cudaMalloc(&d_surface_info, sizeof(SurfaceInfo)));
        CUDA_CHECK(cudaMemcpy(
            d_surface_info,
            &surface_info, sizeof(SurfaceInfo),
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

    AreaEmitter::Data AreaEmitter::getData() const 
    {
        return { m_texture->getData(), m_intensity, m_twosided };
    }

    SurfaceInfo* AreaEmitter::surfaceInfoDevicePtr() const {
        return d_surface_info;
    }

} // namespace prayground