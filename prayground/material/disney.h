#pragma once

#include <prayground/core/material.h>
#include <prayground/core/texture.h>

namespace prayground {

    class Disney final : public Material {
    public:
        struct Data
        {
            Texture::Data base;
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
        };

#ifndef __CUDACC__
        Disney(const SurfaceCallableID& surface_callable_id, 
               const std::shared_ptr<Texture>& base, 
               float subsurface=0.8f, float metallic=0.5f,
               float specular=0.0f, float specular_tint=0.0f,
               float roughness=0.4f, float anisotropic=0.0f, 
               float sheen=0.0f, float sheen_tint=0.5f,
               float clearcoat=0.0f, float clearcoat_gloss=0.0f, bool twosided=true);
        ~Disney();

        SurfaceType surfaceType() const override;
        
        SurfaceInfo surfaceInfo() const override;

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

        void setClearcoatGloss(float clearcoat_gloss);
        float clearcoatGloss() const;

        void setTwosided(bool twosided);
        bool twosided() const;

        Data getData() const;
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

#endif // __CUDACC__
    };

}