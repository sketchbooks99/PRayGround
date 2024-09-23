#pragma once

#include <prayground/core/material.h>
#include <prayground/core/texture.h>

namespace prayground {

    class RoughConductor : public Material
    {
    public:
        struct Data {
            Texture::Data texture;
            float roughness;
            float anisotropic;
            bool twosided;
        };

#ifndef __CUDACC__
        RoughConductor(
            const SurfaceCallableID& callable_id, 
            float roughness = 0.5f, 
            float anisotropic = 0.0f, 
            bool twosided = true);
        
        ~RoughConductor();

        SurfaceType surfaceType() const override;

        void copyToDevice() override;

        void free() override;

        void setTexture(const std::shared_ptr<Texture>& texture) override;
        std::shared_ptr<Texture> texture() const override;

        void setRoughness(const float roughness);
        const float& roughness() const;

        void setAnisotropic(const float anisotropic);
        const float& anisotropic() const;

        Data getData() const;
    private:
        std::shared_ptr<Texture> m_texture;
        float m_roughness;
        float m_anisotropic;
        bool m_twosided;
#endif
    };

} // namespace prayground