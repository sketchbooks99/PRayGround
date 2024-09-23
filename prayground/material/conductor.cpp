#include "conductor.h"

namespace prayground {

    // ------------------------------------------------------------------
    Conductor::Conductor(
        const SurfaceCallableID& surface_callable_id, 
        const std::shared_ptr<Texture>& texture, 
        bool twosided, 
        Thinfilm thinfilm
    )
        : Material(surface_callable_id), 
        m_texture(texture), 
        m_twosided(twosided), 
        m_thinfilm(thinfilm)
    {
    
    }

    Conductor::~Conductor()
    {

    }

    // ------------------------------------------------------------------
    SurfaceType Conductor::surfaceType() const
    {
        return SurfaceType::Reflection;
    }

    // ------------------------------------------------------------------
    void Conductor::copyToDevice()
    {
        Material::copyToDevice();
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();

        m_thinfilm.copyToDevice();

        Data data = this->getData();

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(Data), 
            cudaMemcpyHostToDevice
        ));

        // Copy surface info to the device
        SurfaceInfo surface_info { 
            .data = d_data,
            .callable_id = m_surface_callable_id,
            .type = surfaceType(),
            .use_bumpmap = useBumpmap(),
            .bumpmap = bumpmapData()
        };
        if (!d_surface_info)
            CUDA_CHECK(cudaMalloc(&d_surface_info, sizeof(SurfaceInfo)));
        CUDA_CHECK(cudaMemcpy(
            d_surface_info,
            &surface_info, sizeof(SurfaceInfo),
            cudaMemcpyHostToDevice
        ));
    }

    void Conductor::free()
    {
        m_texture->free();
        Material::free();
    }

    // ------------------------------------------------------------------
    void Conductor::setTexture(const std::shared_ptr<Texture>& texture)
    {
        m_texture = texture;
    }

    std::shared_ptr<Texture> Conductor::texture() const 
    {
        return m_texture;
    }

    void Conductor::setTwosided(bool twosided)
    {
        m_twosided = twosided;
    }

    bool Conductor::twosided() const
    {
        return m_twosided;
    }

    void Conductor::setThinfilm(const Thinfilm& thinfilm)
    {
        m_thinfilm = thinfilm;
    }
    Thinfilm Conductor::thinfilm() const
    {
        return m_thinfilm;
    }

    Conductor::Data Conductor::getData() const
    {
        return Data{
            .texture = m_texture->getData(),
            .twosided = m_twosided,
            .thinfilm = m_thinfilm.getData()
        };
    }

} // namespace prayground