#pragma once

#include <cuda/random.h>
#include "../core/material.h"
#include "../core/bsdf.h"
#include "../core/onb.h"
#include "../core/color.h"
#include "../optix/sbt.h"
#include "../texture/constant.h"

namespace oprt {

struct DisneyData {
    void* base_tex;             // base color
    float subsurface;          
    float metallic;
    float specular;
    float specular_tint;
    float roughness;
    float anisotropic;
    float sheen;
    float sheen_tint;
    float clearcoat;
    float clearcoat_gloss;   
    bool twosided;  
    unsigned int tex_func_idx;  
};

#ifndef __CUDACC__

class Disney final : public Material {
public:
    Disney(){}

    Disney(Texture* base, float subsurface=0.8f, float metallic=0.1f,
           float specular=0.5f, float specular_tint=0.0f,
           float roughness=0.4f, float anisotropic=0.0f, 
           float sheen=0.0f, float sheen_tint=0.5f,
           float clearcoat=0.0f, float clearcoat_gloss=0.0f, bool twosided=true)
    : m_base(base), m_subsurface(subsurface), m_metallic(metallic),
      m_specular(specular), m_specular_tint(specular_tint),
      m_roughness(roughness), m_anisotropic(anisotropic),
      m_sheen(sheen), m_sheen_tint(sheen_tint),
      m_clearcoat(clearcoat), m_clearcoat_gloss(clearcoat_gloss),
      m_twosided(twosided) {}
    
    ~Disney() {}

    void prepare_data() override {
        m_base->prepare_data();

        DisneyData data = {
            m_base->get_dptr(),
            m_subsurface,
            m_metallic, 
            m_specular, 
            m_specular_tint,
            m_roughness,
            m_anisotropic,
            m_sheen,
            m_sheen_tint,
            m_clearcoat,
            m_clearcoat_gloss,
            m_twosided,
            static_cast<unsigned int>(m_base->type()) + static_cast<unsigned int>(MaterialType::Count) * 2
        };

        CUDA_CHECK(cudaMalloc(&d_data, sizeof(DisneyData)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(DisneyData),
            cudaMemcpyHostToDevice
        ));
    }

    MaterialType type() const override { return MaterialType::Disney; }

    void set_subsurface(float subsurface) { m_subsurface = subsurface; }
    float subsurface() const { return m_subsurface; }

    void set_metallic(float metallic) { m_metallic = metallic; }
    float metallic() const { return m_metallic; }

    void set_specular(float specular) { m_specular = specular; }
    float specular() const { return m_specular; }

    void set_specular_tint(float specular_tint) { m_specular_tint = specular_tint; }
    float specular_tint() const { return m_specular_tint; }

    void set_roughness(float roughness) { m_roughness = roughness; }
    float roughness() const { return m_roughness; }

    void set_anisotropic(float anisotropic) { m_anisotropic = anisotropic; }
    float anisotropic() const { return m_anisotropic; }

    void set_sheen(float sheen) { m_sheen = sheen; }
    float sheen() const { return m_sheen; }

    void set_sheen_tint(float sheen_tint) { m_sheen_tint = sheen_tint; }
    float sheen_tint() const { return m_sheen_tint; }

    void set_clearcoat(float clearcoat) { m_clearcoat = clearcoat; }
    float clearcoat() const { return m_clearcoat; }

    void set_clearcoat_gloss(float clearcoat_gloss) { m_clearcoat_gloss = clearcoat_gloss; }
    float clearcoat_gloss() const { return m_clearcoat_gloss; }
private:
    Texture* m_base;
    float m_subsurface;
    float m_metallic;
    float m_specular, m_specular_tint;
    float m_roughness;
    float m_anisotropic;
    float m_sheen, m_sheen_tint;
    float m_clearcoat, m_clearcoat_gloss;
    bool m_twosided;
};

#else

CALLABLE_FUNC void DC_FUNC(sample_disney)(SurfaceInteraction* si, void* matdata)
{
    const DisneyData* disney = reinterpret_cast<DisneyData*>(matdata);

    if (disney->twosided)
        si->n = faceforward(si->n, -si->wi, si->n);

    unsigned int seed = si->seed;
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    const float diffuse_ratio = 1.0f - disney->metallic;
    Onb onb(si->n);

    if (rnd(seed) < diffuse_ratio)
    {
        float3 w_in = cosine_sample_hemisphere(z1, z2);
        onb.inverse_transform(w_in);
        si->wo = normalize(w_in);
    }
    else
    {
        float3 h = sample_ggx(z1, z2, disney->roughness);
        onb.inverse_transform(h);
        si->wo = normalize(reflect(si->wi, h));
    }
    si->seed = seed;
    si->trace_terminate = false;
}

/**
 * @ref: https://rayspace.xyz/CG/contents/Disney_principled_BRDF/
 * 
 * @note 
 * ===== Prefix =====
 * F : fresnel 
 * f : function
 * G : geometry function
 * D : NDF
 */
CALLABLE_FUNC float3 CC_FUNC(bsdf_disney)(SurfaceInteraction* si, void* matdata)
{   
    const DisneyData* disney = reinterpret_cast<DisneyData*>(matdata);
    si->emission = make_float3(0.0f);

    const float3 V = -si->wi;
    const float3 L = si->wo;
    const float3 N = si->n;

    const float NdotV = dot(N, V);
    const float NdotL = dot(N, L);

    if (NdotV <= 0.0f || NdotL <= 0.0f) {
        return make_float3(0.0f);
    }

    const float3 H = normalize(V + L);
    const float NdotH = dot(N, H);
    const float LdotH /* = VdotH */ = dot(L, H);

    const float3 base_color = optixDirectCall<float3, SurfaceInteraction*, void*>(
        disney->tex_func_idx, si, disney->base_tex
    );

    // Diffuse term (diffuse, subsurface, sheen) ======================
    // Diffuse
    const float Fd90 = 0.5f + 2.0f * disney->roughness * LdotH*LdotH;
    const float FVd90 = ft_schlick(NdotV, Fd90);
    const float FLd90 = ft_schlick(NdotL, Fd90);
    const float3 f_diffuse = base_color * FVd90 * FLd90 / M_PIf;

    // Subsurface
    const float Fss90 = disney->roughness * LdotH*LdotH;
    const float FVss90 = ft_schlick(NdotV, Fss90);
    const float FLss90 = ft_schlick(NdotL, Fss90); 
    const float3 f_subsurface = (base_color / M_PIf) * 1.25f * (FVss90 * FLss90 * ((1.0f / (NdotV * NdotL)) - 0.5f) + 0.5f);

    // Sheen
    const float3 rho_tint = base_color / luminance(base_color);
    const float3 rho_sheen = lerp(make_float3(1.0f), rho_tint, disney->sheen_tint); 
    const float3 f_sheen = disney->sheen * rho_sheen * powf(1.0f - LdotH, 5.0f);

    // Specular term (specular, clearcoat) ============================
    // Spcular
    const float3 rho_specular = lerp(make_float3(1.0f), rho_tint, disney->specular_tint);
    const float3 Fs0 = lerp(0.08f * disney->specular * rho_specular, base_color, disney->metallic);
    const float3 FHs0 = fr_schlick(LdotH, Fs0);
    const float alpha = fmaxf(0.001f, disney->roughness * disney->roughness); // Remapping of roughness
    const float Dggx = GTR2(NdotH, alpha);
          float Gggx = geometry_smith(N, V, L, alpha);
    const float3 f_specular = FHs0 * Dggx * Gggx / (4.0f * NdotV * NdotL);

    // Clearcoat
    const float Fcc = fr_schlick(NdotH, 0.04f);
    const float alpha_cc = 0.1f + (0.001f - 0.1f) * disney->clearcoat_gloss; // lerp
    const float DBerry = GTR1(NdotH, alpha_cc);
                Gggx = geometry_smith(N, V, L, 0.25f);
    const float3 f_clearcoat = make_float3( 0.25f * disney->clearcoat * (Fcc * DBerry * Gggx) / (4.0f * NdotV * NdotL) ); 

    const float3 out = ( 1.0f - disney->metallic ) * ( lerp( f_diffuse, f_subsurface, disney->subsurface ) + f_sheen ) + f_specular + f_clearcoat;
    return out * clamp(NdotL, 0.0f, 1.0f);    
}

/**
 * @ref (temporal) http://simon-kallweit.me/rendercompo2015/report/#adaptivesampling
 * 
 * @todo Search and consider correct evaluation of PDF.
 */
CALLABLE_FUNC float DC_FUNC(pdf_disney)(SurfaceInteraction* si, void* matdata)
{
    const DisneyData* disney = reinterpret_cast<DisneyData*>(matdata);

    const float3 V = -si->wi;
    const float3 L = si->wo;
    const float3 N = si->n;

    const float NdotV = fmaxf(0.0f, dot(N, V));
    const float NdotL = fmaxf(0.0f, dot(N, L));

    const float alpha = fmaxf(0.001f, disney->roughness * disney->roughness);
    const float alpha_cc = 0.1f + (0.001f - 0.1f) * disney->clearcoat_gloss; // lerp
    const float3 H = normalize(V + L);
    const float NdotH = fmaxf(0.0f, dot(H, N));

    const float pdf_Dggx = GTR2(NdotH, alpha);
    const float pdf_DBerry = GTR1(NdotH, alpha_cc);
    const float ratio = 1.0f / (1.0f + disney->clearcoat);
    const float pdf_specular = (pdf_Dggx + ratio * (pdf_DBerry - pdf_Dggx)) / (4.0f * NdotV * NdotL);
    const float pdf_diffuse = NdotL / M_PIf;

    return (1.0f - disney->metallic) * pdf_diffuse + disney->metallic * pdf_specular;
}

#endif

}