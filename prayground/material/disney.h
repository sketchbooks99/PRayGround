#pragma once

// For cuda

#include <prayground/core/material.h>
#include <prayground/texture/constant.h>

namespace prayground {

struct DisneyData {
    void* base_tex_data;             // base color
    float subsurface;          
    float metallic;
    float specular;
    float specular_tint;
    float roughness;
    float anisotropic;
    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;   
    bool twosided;  
    unsigned int base_program_id;  
};

#ifndef __CUDACC__

class Disney final : public Material {
public:
    Disney(){}

    Disney(const std::shared_ptr<Texture>& base, float subsurface=0.8f, float metallic=0.1f,
           float specular=0.0f, float specular_tint=0.0f,
           float roughness=0.4f, float anisotropic=0.0f, 
           float sheen=0.0f, float sheen_tint=0.5f,
           float clearcoat=0.0f, float clearcoat_gloss=0.0f, bool twosided=true)
    : m_base(base), m_subsurface(subsurface), m_metallic(metallic),
      m_specular(specular), m_specular_tint(specular_tint),
      m_roughness(roughness), m_anisotropic(anisotropic),
      m_sheen(sheen), m_sheen_tint(sheen_tint),
      m_clearcoat(clearcoat), m_clearcoat_gloss(clearcoat_gloss),
      m_twosided(twosided) {}
    
    ~Disney() {}

    void copyToDevice() override {
        if (!m_base->devicePtr())
            m_base->copyToDevice();

        DisneyData data = {
            m_base->devicePtr(),
            m_subsurface,
            m_metallic, 
            m_specular, 
            m_specular_tint,
            m_roughness,
            m_anisotropic,
            m_sheen,
            m_sheen_tint,
            m_clearcoat,
            m_clearcoat_gloss,
            m_twosided,
            m_base->programId()
        };

        CUDA_CHECK(cudaMalloc(&d_data, sizeof(DisneyData)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(DisneyData),
            cudaMemcpyHostToDevice
        ));
    }

    void free() override
    {
        m_base->free();
    }

    void setBaseTexture(const std::shared_ptr<Texture>& base) { m_base = base; }
    std::shared_ptr<Texture> base() const { return m_base; }

    void setSubsurface(float subsurface) { m_subsurface = subsurface; }
    float subsurface() const { return m_subsurface; }

    void setMetallic(float metallic) { m_metallic = metallic; }
    float metallic() const { return m_metallic; }

    void setSpecular(float specular) { m_specular = specular; }
    float specular() const { return m_specular; }

    void setSpecularTint(float specular_tint) { m_specular_tint = specular_tint; }
    float specularTint() const { return m_specular_tint; }

    void setRoughness(float roughness) { m_roughness = roughness; }
    float roughness() const { return m_roughness; }

    void setAnisotropic(float anisotropic) { m_anisotropic = anisotropic; }
    float anisotropic() const { return m_anisotropic; }

    void setSheen(float sheen) { m_sheen = sheen; }
    float sheen() const { return m_sheen; }

    void setSheenTint(float sheen_tint) { m_sheen_tint = sheen_tint; }
    float sheenTint() const { return m_sheen_tint; }

    void setClearcoat(float clearcoat) { m_clearcoat = clearcoat; }
    float clearcoat() const { return m_clearcoat; }

    void setClearoatGloss(float clearcoat_gloss) { m_clearcoat_gloss = clearcoat_gloss; }
    float clearcoatGloss() const { return m_clearcoat_gloss; }
private:
    std::shared_ptr<Texture> m_base;
    float m_subsurface;
    float m_metallic;
    float m_specular, m_specular_tint;
    float m_roughness;
    float m_anisotropic;
    float m_sheen, m_sheen_tint;
    float m_clearcoat, m_clearcoat_gloss;
    bool m_twosided;
    
};

#else

#endif

}