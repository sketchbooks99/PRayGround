#pragma once

// For cuda

#include <prayground/core/material.h>

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
    Disney(const std::shared_ptr<Texture>& base, 
           float subsurface=0.8f, float metallic=0.1f,
           float specular=0.0f, float specular_tint=0.0f,
           float roughness=0.4f, float anisotropic=0.0f, 
           float sheen=0.0f, float sheen_tint=0.5f,
           float clearcoat=0.0f, float clearcoat_gloss=0.0f, bool twosided=true);
    ~Disney();

    void copyToDevice() override;
    void free() override;

    void setBaseTexture(const std::shared_ptr<Texture>& base);
    std::shared_ptr<Texture> base() const;

    void setSubsurface(float subsurface);
    float subsurface() const;

    void setMetallic(float metallic);
    float metallic() const;

    void setSpecular(float specular);
    float specular() const;

    void setSpecularTint(float specular_tint);
    float specularTint() const;

    void setRoughness(float roughness);
    float roughness() const;

    void setAnisotropic(float anisotropic);
    float anisotropic() const;

    void setSheen(float sheen);
    float sheen() const;

    void setSheenTint(float sheen_tint);
    float sheenTint() const;

    void setClearcoat(float clearcoat);
    float clearcoat() const;

    void setClearoatGloss(float clearcoat_gloss);
    float clearcoatGloss() const;
    
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