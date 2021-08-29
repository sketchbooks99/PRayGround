/**
 * @brief Perfect smooth conductor material
 */

#pragma once

#include <prayground/core/material.h>

namespace prayground {

struct ConductorData {
    void* tex_data;
    bool twosided;
    uint32_t tex_program_id;
};

#ifndef __CUDACC__
class Conductor final : public Material {
public:
    Conductor(const std::shared_ptr<Texture>& texture, bool twosided=true);
    ~Conductor();

    void copyToDevice() override;
    void free() override;

    void setTexture(const std::shared_ptr<Texture>& texture);
    std::shared_ptr<Texture> texture() const;

private:
    std::shared_ptr<Texture> m_texture;
    bool m_twosided;
};

#endif

}



