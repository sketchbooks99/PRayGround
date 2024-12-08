#include "disney.h"

namespace prayground {

    // ------------------------------------------------------------------
    Disney::Disney(
        const SurfaceCallableID& surface_callable_id,
        const std::shared_ptr<Texture>& albedo, 
        float subsurface, float metallic,
        float specular, float specular_tint, 
        float roughness, float anisotropic, 
        float sheen, float sheen_tint, 
        float clearcoat, float clearcoat_gloss, bool twosided, 
        Thinfilm thinfilm)
    : Material(surface_callable_id), 
      m_albedo(albedo), 
      m_subsurface(subsurface), m_metallic(metallic),
      m_specular(specular), m_specular_tint(specular_tint),
      m_roughness(roughness), m_anisotropic(anisotropic), 
      m_sheen(sheen), m_sheen_tint(sheen_tint),
      m_clearcoat(clearcoat), m_clearcoat_gloss(clearcoat_gloss), 
      m_twosided(twosided), m_thinfilm(thinfilm)
    {

    }

    Disney::~Disney()
    {

    }

    // ------------------------------------------------------------------
    SurfaceType Disney::surfaceType() const 
    {
        return SurfaceType::RoughReflection;
    }

    // ------------------------------------------------------------------
    void Disney::copyToDevice()
    {
        Material::copyToDevice();
        if (!m_albedo->devicePtr())
            m_albedo->copyToDevice();

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

    void Disney::free() 
    {
        m_albedo->free();
        Material::free();
    }

    // ------------------------------------------------------------------
    void Disney::setTexture(const std::shared_ptr<Texture>& albedo)
    {
        m_albedo = albedo;
    }
    std::shared_ptr<Texture> Disney::texture() const 
    {
        return m_albedo;
    }

    // ------------------------------------------------------------------
    void Disney::setSubsurface(float subsurface) 
    { 
        m_subsurface = subsurface; 
    }
    float Disney::subsurface() const 
    { 
        return m_subsurface; 
    }

    // ------------------------------------------------------------------
    void Disney::setMetallic(float metallic) 
    { 
        m_metallic = metallic; 
    }
    float Disney::metallic() const 
    { 
        return m_metallic; 
    }

    // ------------------------------------------------------------------
    void Disney::setSpecular(float specular) 
    { 
        m_specular = specular; 
    }
    float Disney::specular() const 
    { 
        return m_specular; 
    }

    // ------------------------------------------------------------------
    void Disney::setSpecularTint(float specular_tint) 
    { 
        m_specular_tint = specular_tint; 
    }
    float Disney::specularTint() const 
    { 
        return m_specular_tint; 
    }

    // ------------------------------------------------------------------
    void Disney::setRoughness(float roughness) 
    { 
        m_roughness = roughness; 
    }
    float Disney::roughness() const 
    { 
        return m_roughness; 
    }

    // ------------------------------------------------------------------
    void Disney::setAnisotropic(float anisotropic) 
    { 
        m_anisotropic = anisotropic; 
    }
    float Disney::anisotropic() const 
    { 
        return m_anisotropic; 
    }

    // ------------------------------------------------------------------
    void Disney::setSheen(float sheen) 
    { 
        m_sheen = sheen; 
    }
    float Disney::sheen() const 
    { 
        return m_sheen; 
    }

    // ------------------------------------------------------------------
    void Disney::setSheenTint(float sheen_tint) 
    { 
        m_sheen_tint = sheen_tint; 
    }
    float Disney::sheenTint() const 
    { 
        return m_sheen_tint; 
    }

    // ------------------------------------------------------------------
    void Disney::setClearcoat(float clearcoat) 
    { 
        m_clearcoat = clearcoat; 
    }
    float Disney::clearcoat() const 
    { 
        return m_clearcoat; 
    }

    // ------------------------------------------------------------------
    void Disney::setClearcoatGloss(float clearcoat_gloss) 
    { 
        m_clearcoat_gloss = clearcoat_gloss; 
    }
    float Disney::clearcoatGloss() const 
    { 
        return m_clearcoat_gloss; 
    }

    void Disney::setTwosided(bool twosided)
    {
        m_twosided = twosided;
    }

    bool Disney::twosided() const
    {
        return m_twosided;
    }

    void Disney::setThinfilm(const Thinfilm& thinfilm)
    {
        m_thinfilm = thinfilm;
    }
    Thinfilm Disney::thinfilm() const
    {
        return m_thinfilm;
    }

    Disney::Data Disney::getData() const
    {
        return Data{
            .albedo = m_albedo->getData(),
            .subsurface = m_subsurface,
            .metallic = m_metallic,
            .specular = m_specular,
            .specular_tint = m_specular_tint,
            .roughness = m_roughness,
            .anisotropic = m_anisotropic,
            .sheen = m_sheen,
            .sheen_tint = m_sheen_tint,
            .clearcoat = m_clearcoat,
            .clearcoat_gloss = m_clearcoat_gloss,
            .twosided = m_twosided, 
            .thinfilm = m_thinfilm.getData()
        };
    }

} // namespace prayground