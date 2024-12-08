/**
 * @brief smooth conductor material
 */

#pragma once

#include <prayground/core/material.h>
#include <prayground/core/texture.h>
#include "thinfilm.h"

namespace prayground {

    class Conductor final : public Material {
    public:
        struct Data {
            Texture::Data texture;
            bool twosided;

            // Thinfilm
            bool use_thinfilm;
            Thinfilm::Data thinfilm;
        };

#ifndef __CUDACC__
        Conductor(
            const SurfaceCallableID& surface_callable_id, 
            const std::shared_ptr<Texture>& texture, 
            bool twosided=true, 
            Thinfilm thinfilm = Thinfilm()
        );
        ~Conductor();

        SurfaceType surfaceType() const override;

        void copyToDevice() override;
        void free() override;

        void setTexture(const std::shared_ptr<Texture>& texture);
        std::shared_ptr<Texture> texture() const;

        void setTwosided(bool twosided);
        bool twosided() const;

        void setThinfilm(const Thinfilm& thinfilm);
        Thinfilm thinfilm() const;

        Data getData() const;
    private:
        std::shared_ptr<Texture> m_texture;
        bool m_twosided;

        Thinfilm m_thinfilm;
#endif
    };

}



