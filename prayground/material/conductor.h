/**
 * @brief smooth conductor material
 */

#pragma once

#include <prayground/core/material.h>
#include <prayground/core/texture.h>

namespace prayground {

    class Conductor final : public Material {
    public:
        struct Data {
            Texture::Data texture;
            bool twosided;

            // Fresnel
            Vec3f ior;

            // Thin film
            float tf_thickness;
            float tf_ior;
            Vec3f extinction;
        };

#ifndef __CUDACC__
        Conductor(
            const SurfaceCallableID& surface_callable_id, 
            const std::shared_ptr<Texture>& texture, 
            bool twosided=true, 
            Vec3f ior = Vec3f(1.0f),
            float tf_thickness=0.0f,
            float tf_ior=0.0f, 
            Vec3f extinction = Vec3f(0.0f)
        );
        ~Conductor();

        SurfaceType surfaceType() const override;

        SurfaceInfo surfaceInfo() const override;

        void copyToDevice() override;
        void free() override;

        void setTexture(const std::shared_ptr<Texture>& texture);
        std::shared_ptr<Texture> texture() const;

        void setTwosided(bool twosided);
        bool twosided() const;

        void setIOR(const Vec3f& ior);
        Vec3f ior() const;

        void setThinfilmThickness(float tf_thickness);
        float thinfilmThickness() const;

        void setThinfilmIOR(float tf_ior);
        float thinfilmIOR() const;

        void setExtinction(const Vec3f& extinction);
        Vec3f extinction() const;

        Data getData() const;
    private:
        std::shared_ptr<Texture> m_texture;
        bool m_twosided;

        // Fresnel
        Vec3f m_ior;

        // Thin film
        float m_tf_thickness;
        float m_tf_ior;
        Vec3f m_extinction;
#endif
    };

}



