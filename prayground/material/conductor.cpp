#include "conductor.h"

namespace prayground {

    // ------------------------------------------------------------------
    Conductor::Conductor(const SurfaceCallableID& surface_callable_id, const std::shared_ptr<Texture>& texture, bool twosided)
        : Material(surface_callable_id), m_texture(texture), m_twosided(twosided)
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

    SurfaceInfo Conductor::surfaceInfo() const
    {
        ASSERT(d_data, "Material data on device hasn't been allocated yet.");

        return SurfaceInfo{
            .data = d_data,
            .callable_id = m_surface_callable_id,
            .type = SurfaceType::RoughReflection
        };
    }

    // ------------------------------------------------------------------
    void Conductor::copyToDevice()
    {
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();

        Data data = this->getData();

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(Data), 
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

    Conductor::Data Conductor::getData() const
    {
        return Data{
            .texture = m_texture->getData(),
            .twosided = m_twosided
        };
    }

} // namespace prayground