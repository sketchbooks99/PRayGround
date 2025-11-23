#include "roughconductor.h"

namespace prayground {

    RoughConductor::RoughConductor(const SurfaceCallableID& callable_id, float roughness, float anisotropic, bool twosided)
        : Material(callable_id), m_roughness(roughness), m_anisotropic(anisotropic), m_twosided(twosided)
    {
    }

    RoughConductor::~RoughConductor()
    {
    }

    SurfaceType RoughConductor::surfaceType() const
    {
        return SurfaceType::RoughReflection;
    }

    void RoughConductor::copyToDevice()
    {
        if (!m_texture->devicePtr())
            m_texture->copyToDevice();

        Data data = this->getData();

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));

        CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(Data), cudaMemcpyHostToDevice));

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

    void RoughConductor::free()
    {
        m_texture->free();
        Material::free();
    }

    void RoughConductor::setTexture(const std::shared_ptr<Texture>& texture)
    {
        m_texture = texture;
    }

    std::shared_ptr<Texture> RoughConductor::texture() const
    {
        return m_texture;
    }
    void RoughConductor::setRoughness(const float roughness)
    {
        m_roughness = roughness;
    }

    const float& RoughConductor::roughness() const
    {
        return m_roughness;
    }

    void RoughConductor::setAnisotropic(const float anisotropic)
    {
        m_anisotropic = anisotropic;
    }
    const float& RoughConductor::anisotropic() const
    {
        return m_anisotropic;
    }

    RoughConductor::Data RoughConductor::getData() const
    {
        return Data{
            .texture = m_texture->getData(),
            .roughness = m_roughness, 
            .anisotropic = m_anisotropic, 
            .twosided = m_twosided
        };
    }
} // namespace prayground