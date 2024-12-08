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
   
    void Diffuse::setTexture(const std::shared_ptr<Texture>& texture)
    {
        m_texture = texture;
    }

    std::shared_ptr<Texture> Diffuse::texture() const
    {
        return m_texture;
    }

    // ------------------------------------------------------------------
    void Diffuse::copyToDevice()
    {
        Material::copyToDevice();
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

        // Copy surface info to the device
        SurfaceInfo surface_info{
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