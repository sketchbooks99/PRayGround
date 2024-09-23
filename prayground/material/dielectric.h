#pragma once 

#include <prayground/core/material.h>
#include <prayground/core/texture.h>
#include "thinfilm.h"

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
            // Dielectric data
            Texture::Data texture;
            float ior;
            float absorb_coeff; 
            Sellmeier sellmeier;

            // Thinfilm
            Thinfilm::Data thinfilm;
        };

#ifndef __CUDACC__
        Dielectric(
            const SurfaceCallableID& surface_callable_id, 
            const std::shared_ptr<Texture>& texture, 
            float ior, 
            float absorb_coeff = 0.0f, 
            Sellmeier sellmeier = Sellmeier::None, 
            Thinfilm thinfilm = Thinfilm()
        );
        ~Dielectric();

        SurfaceType surfaceType() const override;

        void copyToDevice() override;
        void free() override;

        void setIor(const float ior);
        float ior() const;

        void setAbsorbCoeff(const float absorb_coeff);
        float absorbCoeff() const;

        void setSellmeierType(Sellmeier ior_func);
        Sellmeier sellmeierType() const;

        void setTexture(const std::shared_ptr<Texture>& texture);
        std::shared_ptr<Texture> texture() const;

        void setThinfilm(const Thinfilm& thinfilm);
        Thinfilm thinfilm() const;

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
        Thinfilm m_thinfilm;
#endif
    };

}