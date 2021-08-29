#pragma once

#include <prayground/core/material.h>
#include <prayground/core/texture.h>

namespace prayground {

struct DiffuseData {
    void* tex_data;
    bool twosided;
    unsigned int tex_program_id;
};

#ifndef __CUDACC__

class Diffuse final : public Material {
public:
    Diffuse(const std::shared_ptr<Texture>& texture, bool twosided=true);
    ~Diffuse();

    void copyToDevice() override;
    void free() override;
private:
    std::shared_ptr<Texture> m_texture;
    bool m_twosided;
};

#else

#endif

}