/**
 * @brief Perfect smooth conductor material
 */

#pragma once

#include <prayground/core/material.h>
#include <prayground/core/texture.h>

namespace prayground {

class Conductor final : public Material {
public:
    struct Data {
        Texture::Data tex_data;
        bool twosided;
    };

#ifndef __CUDACC__
    Conductor(const std::shared_ptr<Texture>& texture, bool twosided=true);
    ~Conductor();

    SurfaceType surfaceType() const override;

    void copyToDevice() override;
    void free() override;

    void setTexture(const std::shared_ptr<Texture>& texture);
    std::shared_ptr<Texture> texture() const;
private:
    std::shared_ptr<Texture> m_texture;
    bool m_twosided;

#endif
};

}



