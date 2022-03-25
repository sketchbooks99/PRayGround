#pragma once

#include <prayground/core/material.h>
#include <prayground/core/texture.h>

namespace prayground {

class Diffuse final : public Material {
public:
    struct Data {
        Texture::Data texture;
        bool twosided;
    };

#ifndef __CUDACC__
    Diffuse(const std::shared_ptr<Texture>& texture, bool twosided=true);
    ~Diffuse();

    SurfaceType surfaceType() const override;

    void copyToDevice() override;
    void free() override;
private:
    std::shared_ptr<Texture> m_texture;
    bool m_twosided;
#endif
};

}