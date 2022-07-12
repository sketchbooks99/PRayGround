#pragma once 

#include <prayground/core/material.h>
#include <prayground/core/texture.h>

namespace prayground {

enum class Sellmeier
{
    None, 
    BK7, 
    Diamond
};

class Dielectric final : public Material {
public:
    struct Data {
        Texture::Data texture;
        float ior;
        float absorb_coeff; 
        Sellmeier sellmeier;
    };

#ifndef __CUDACC__
    Dielectric(
        const SurfaceCallableID& surface_callable_id, 
        const std::shared_ptr<Texture>& texture, float ior, 
        float absorb_coeff = 0.0f, Sellmeier sellmeier = Sellmeier::None);
    ~Dielectric();

    SurfaceType surfaceType() const override;

    SurfaceInfo surfaceInfo() const override;

    void copyToDevice() override;
    void free() override;

    void setIor(const float ior);
    float ior() const;

    void setAbsorbCoeff(const float absorb_coeff);
    float absorbCoeff() const;

    void setSellmeierType(Sellmeier ior_func);

    void setTexture(const std::shared_ptr<Texture>& texture);
    std::shared_ptr<Texture> texture() const;

    Data getData() const;
private:
    std::shared_ptr<Texture> m_texture;
    float m_ior;
    /// For controlling absorption intensity according to the distance
    /// that the ray travel inside of the glass. 
    float m_absorb_coeff; 
    Sellmeier m_sellmeier;

#endif
};

}