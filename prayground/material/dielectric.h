#pragma once 

#include <prayground/core/material.h>

namespace prayground {

struct DielectricData {
    void* tex_data;
    float ior;
    unsigned int tex_program_id;
};

#ifndef __CUDACC__

class Dielectric final : public Material {
public:
    Dielectric(const std::shared_ptr<Texture>& texture, float ior);
    ~Dielectric();

    void copyToDevice() override;
    void free() override;

    void setIor(const float ior);
    float ior() const;

    void setTexture(const std::shared_ptr<Texture>& texture);
    std::shared_ptr<Texture> texture() const;
private:
    std::shared_ptr<Texture> m_texture;
    float m_ior;
};

#else 

#endif

}