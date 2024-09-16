#pragma once 

#include <prayground/core/material.h>
#include <prayground/core/texture.h>

namespace prayground {

    enum class Sellmeier : uint8_t
    {
        None, 
        BK7, 
        Diamond
    };

    class Dielectric final : public Material {
    public:
        struct Data {
            // Base material data
            Material::Data base;

            // Dielectric data
            Texture::Data texture;
            float ior;
            float absorb_coeff; 
            Sellmeier sellmeier;

            // Thin film
            Texture::Data tf_thickness;
            float tf_ior;
            Vec3f extinction;
        };

#ifndef __CUDACC__
        Dielectric(
            const SurfaceCallableID& surface_callable_id, 
            const std::shared_ptr<Texture>& texture, 
            float ior, 
            float absorb_coeff = 0.0f, 
            Sellmeier sellmeier = Sellmeier::None, 
            const std::shared_ptr<Texture>& thickness = nullptr, 
            float tf_ior = 0.0f, 
            Vec3f extinction = Vec3f(0.0f)
        );
        ~Dielectric();

        SurfaceType surfaceType() const override;

        SurfaceInfo surfaceInfo() const override;

        void copyToDevice() override;
        void free() override;

        void setIor(const float ior);
        float ior() const;

        void setAbsorbCoeff(const float absorb_coeff);
        float absorbCoeff() const;

        void setSellmeierType(Sellmeier ior_func);
        Sellmeier sellmeierType() const;
        
        void setThinfilmThickness(const std::shared_ptr<Texture>& tf_thickness);
        std::shared_ptr<Texture> thinfilmThickness() const;

        void setThinfilmIOR(const float tf_ior);
        float thinfilmIOR() const;

        void setExtinction(const Vec3f& extinction);
        Vec3f extinction() const;

        void setTexture(const std::shared_ptr<Texture>& texture);
        std::shared_ptr<Texture> texture() const;

        Data getData() const;
    private:
        std::shared_ptr<Texture> m_texture;
        float m_ior;

        /* 
        For controlling absorption intensity according to the distance 
        that the ray travel inside of the glass. 
        */
        float m_absorb_coeff; 

        /* 
        The type to specify the sellmeier function on the index calculation. 
        This is mainly used for spectrum rendering.
        */
        Sellmeier m_sellmeier;

        // Thin film
        std::shared_ptr<Texture> m_tf_thickness;
        float m_tf_ior;
        Vec3f m_extinction;
#endif
    };

}