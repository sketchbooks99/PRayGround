#pragma once

#include <prayground/core/material.h>
#include <prayground/core/texture.h>
#include <prayground/material/dielectric.h>

namespace prayground {

    class RoughDielectric : public Material
    {
    public:
        struct Data {
            Texture::Data texture;
            float ior;
            float roughness;
            float absorb_coeff;
            Sellmeier sellmeier;
        };

#ifndef __CUDACC__
        RoughDielectric(
            const SurfaceCallableID& callable_id,
            float ior,
            float roughness = 0.5f,
            float absorb_coeff = 0.0f,
            Sellmeier sellmeier = Sellmeier::None);
        
        ~RoughDielectric();

        SurfaceType surfaceType() const override;

        SurfaceInfo surfaceInfo() const override;

        void copyToDevice() override;

        void free() override;

        void setTexture(const std::shared_ptr<Texture>& texture);
        std::shared_ptr<Texture> texture() const;

        void setRoughness(const float roughness);
        const float& roughness() const;

        void setIor(const float ior);
        const float& ior() const;

        void setAbsorbCoeff(const float absorb_coeff);
        const float& absorbCoeff() const;

        Data getData() const;
    private:
        std::shared_ptr<Texture> m_texture;
        float m_ior;
        float m_roughness;

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
#endif
    };

} // namespace prayground