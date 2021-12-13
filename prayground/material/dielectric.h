#pragma once 

#include <prayground/core/material.h>
#include <prayground/core/texture.h>

namespace prayground {

struct DielectricData {
    void* tex_data;
    float ior;
    float absorb_coeff;
    unsigned int tex_program_id;
};

#ifndef __CUDACC__

class Dielectric final : public Material {
public:
    Dielectric(const std::shared_ptr<Texture>& texture, float ior, float absorb_coeff = 0.0f);
    ~Dielectric();

    SurfaceType surfaceType() const override;

    void copyToDevice() override;
    void free() override;

    void setIor(const float ior);
    float ior() const;

    void setAbsorbCoeff(const float absorb_coeff);
    float absorbCoeff() const;

    void setTexture(const std::shared_ptr<Texture>& texture);
    std::shared_ptr<Texture> texture() const;
private:
    std::shared_ptr<Texture> m_texture;
    float m_ior;
    /// For controlling absorption intensity according to the distance
    /// that the ray travel inside of the glass. 
    float m_absorb_coeff; 
};

#else 

#endif

}