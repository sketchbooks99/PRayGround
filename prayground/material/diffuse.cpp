#include "diffuse.h"

namespace prayground {

    // ------------------------------------------------------------------
    Diffuse::Diffuse(const SurfaceCallableID& surface_callable_id, const std::shared_ptr<Texture>& texture, bool twosided)
        : Material(surface_callable_id), m_texture(texture), m_twosided(twosided)
    {

    }

    Diffuse::~Diffuse()
    {

    }

    // ------------------------------------------------------------------
    SurfaceType Diffuse::surfaceType() const 
    {
        return SurfaceType::Diffuse;
    }
    
    SurfaceInfo Diffuse::surfaceInfo() const
    {
        ASSERT(d_data, "Material data on device hasn't been allocated yet.");

        return SurfaceInfo{
            .data = d_data,
            .callable_id = m_surface_callable_id,
            .type = SurfaceType::Diffuse
        };
    }

    // ------------------------------------------------------------------
    void Diffuse::copyToDevice()
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

    void Diffuse::free()
    {
        m_texture->free();
        Material::free();
    }

    Diffuse::Data Diffuse::getData() const
    {
        return Data{
            .texture = m_texture->getData(),
            .twosided = m_twosided
        };
    }
} // namespace prayground