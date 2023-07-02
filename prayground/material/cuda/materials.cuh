#pragma once

// Utility include
#include <prayground/math/vec.h>
#include <prayground/core/bsdf.h>
#include <prayground/core/interaction.h>
#include <prayground/core/onb.h>
#include <prayground/core/sampler.h>
#include <prayground/core/spectrum.h>
#include <prayground/optix/cuda/device_util.cuh>

// Material include
#include <prayground/material/conductor.h>
#include <prayground/material/roughconductor.h>
#include <prayground/material/dielectric.h>
#include <prayground/material/roughdielectric.h>
#include <prayground/material/diffuse.h>
#include <prayground/material/disney.h>

namespace prayground {

    // ----------------------------------------------------------------------------------------
    // Conductor
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec3f pgSamplingSmoothConductor(
        const Conductor::Data* conductor, 
        const Vec3f& wo, 
        Shading& shading)
    {
        if (conductor->twosided)
            shading.n = faceforward(shading.n, -wo, shading.n);
        return reflect(wo, shading.n);
    }

    // ----------------------------------------------------------------------------------------
    // RoughConductor
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec3f pgImportanceSamplingRoughConductor(
        const RoughConductor::Data* roughconductor,
        const Vec3f& wo,
        Shading& shading,
        uint32_t& seed)
    {
        return Vec3f(0, 1, 0);
    }

    INLINE DEVICE Vec3f pgGetRoughConductorBRDF(
        const RoughConductor::Data* roughconductor,
        const Vec3f& wo, const Vec3f& wi,
        const Shading& shading,
        uint32_t& seed)
    {
        return Vec3f(0);
    }

    INLINE DEVICE float pgGetRoughConductorPDF(
        const RoughConductor::Data* roughconductor,
        const Vec3f& wo, const Vec3f& wi,
        const Shading& shading,
        uint32_t& seed)
    {
        return 1.0f;
    }

    // ----------------------------------------------------------------------------------------
    // Dielectric
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec3f pgSamplingSmoothDielectric(
        const Dielectric::Data* dielectric,
        const Vec3f& wo,
        Shading& shading,
        uint32_t& seed)
    {
        float ni = 1.000292f;       /// @todo Consider IOR of current medium where ray goes on
        float nt = dielectric->ior;
        float cosine = dot(wo, shading.n);
        // Check where the ray is going outside or inside
        bool into = cosine < 0;
        Vec3f outward_normal = into ? shading.n : -shading.n;

        // Swap IOR based on ray location
        if (!into) swap(ni, nt);

        // Check if the ray can be refracted
        cosine = fabs(cosine);
        float sine = sqrtf(1.0f - pow2(cosine));
        bool cannot_refract = (ni / nt) * sine > 1.0f;

        // Get reflectivity by the Fresnel equation
        float reflect_prob = fresnel(cosine, ni, nt);
        // Get out going direction of the ray
        if (cannot_refract || reflect_prob > rnd(seed))
            return reflect(wo, outward_normal);
        else
            return refract(wo, outward_normal, cosine, ni, nt);
    }

    // ----------------------------------------------------------------------------------------
    // Diffuse
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec3f pgImportanceSamplingDiffuse(
        const Diffuse::Data* diffuse,
        const Vec3f& wo, Shading& shading, uint32_t& seed
    )
    {
        if (diffuse->twosided)
            shading.n = faceforward(shading.n, -wo, shading.n);

        // Importance sampling in cosine direction on hemisphere
        Vec2f u = UniformSampler::get2D(seed);

        Vec3f wi = cosineSampleHemisphere(u[0], u[1]);
        Onb onb(shading.n);
        onb.inverseTransform(wi);
        return wi;
    }

    INLINE DEVICE float pgGetDiffuseBRDF(const Vec3f& wi, const Vec3f& n)
    {
        return fmaxf(0.0f, dot(n, wi));
    }

    INLINE DEVICE float pgGetDiffusePDF(const Vec3f& wi, const Vec3f& n)
    {
        return fmaxf(0.0f, dot(n, wi));
    }

    // ----------------------------------------------------------------------------------------
    // Disney
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec3f pgImportanceSamplingDisney(
        const Disney::Data* disney, 
        const Vec3f& wo, Shading& shading, 
        uint32_t& seed
    )
    {
        if (disney->twosided)
            shading.n = faceforward(shading.n, -wo, shading.n);

        const Vec2f u = UniformSampler::get2D(seed);
        const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
        Onb onb(shading.n);

        // Importance sampling on cosine direction
        if (rnd(seed) < diffuse_ratio)
        {
            Vec3f wi = cosineSampleHemisphere(u[0], u[1]);
            onb.inverseTransform(wi);
            return normalize(wi);
        }
        // Importance sampling with GGX formula
        else
        {
            float gtr2_ratio = 1.0f / (1.0f + disney->clearcoat);
            Vec3f h;
            const float alpha = fmaxf(0.001f, disney->roughness);
            const float alpha_cc = lerp(0.1f, 0.001f, disney->clearcoat_gloss);
            // Switch sampling function of microfacet normal
            if (rnd(seed) < gtr2_ratio)
            {
                const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
                const float ax = fmaxf(0.001f, pow2(alpha) / aspect);
                const float ay = fmaxf(0.001f, pow2(alpha) * aspect);
                h = sampleGGXAniso(ax, ay, u[0], u[1]);
            }
            else
            {
                h = sampleGTR1(u[0], u[1], alpha_cc);
            }
            onb.inverseTransform(h);
            return normalize(reflect(wo, h));
        }
    }

    INLINE DEVICE Vec3f pgGetDisneyBRDF(
        const Disney::Data* disney,
        const Vec3f& wo, const Vec3f& wi,
        const Shading& shading,
        const Vec3f& base)
    {
        // V ... View vector, L ... Light vector, N ... Normal
        const Vec3f V = -wo;
        const Vec3f L = wi;
        const Vec3f N = shading.n;

        const float NdotV = dot(N, V);
        const float NdotL = dot(N, L);

        if (NdotL <= 0.0f || NdotV <= 0.0f)
            return Vec3f(0.0f);

        const Vec3f H = (V + L) / 2.0f;
        const float NdotH = dot(N, H);
        const float LdotH /* = VdotH */ = dot(L, H);

        // Diffuse
        const float Fd90 = 0.5f + 2.0f * disney->roughness * pow2(LdotH);
        const float FVd90 = fresnelSchlickT(NdotV, Fd90);
        const float FLd90 = fresnelSchlickT(NdotL, Fd90);
        Vec3f f_diffuse = (base / math::pi) * FVd90 * FLd90;

        // Subsurface
        const float Fss90 = disney->roughness * LdotH * LdotH;
        const float FVss90 = fresnelSchlickT(NdotV, Fss90);
        const float FLss90 = fresnelSchlickT(NdotL, Fss90);
        Vec3f f_subsurface = (base / math::pi) * 1.25f * (FVss90 * FLss90 * ((1.0f / (NdotV * NdotL)) - 0.5f) + 0.5f);

        // Sheen
        const Vec3f rho_tint = base / luminance(base);
        const Vec3f rho_sheen = lerp(Vec3f(1.0f), rho_tint, disney->sheen_tint);
        Vec3f f_sheen = disney->sheen * rho_sheen * powf(1.0f - LdotH, 5.0f);

        // Spcular
        const Vec3f X = shading.dpdu;
        const Vec3f Y = shading.dpdv;
        const float alpha = fmaxf(0.001f, disney->roughness);
        const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
        const float ax = fmaxf(0.001f, pow2(alpha) / aspect);
        const float ay = fmaxf(0.001f, pow2(alpha) * aspect);
        const Vec3f rho_specular = lerp(Vec3f(1.0f), rho_tint, disney->specular_tint);
        const Vec3f Fs0 = lerp(0.08f * disney->specular * rho_specular, base, disney->metallic);
        const Vec3f FHs0 = fresnelSchlickR(LdotH, Fs0);
        const float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
        float Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
              Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);
        Vec3f f_specular = FHs0 * Ds * Gs;

        // Clearcoat
        const float Fcc = fresnelSchlickR(LdotH, 0.04f);
        const float alpha_cc = lerp(0.1f, 0.001f, disney->clearcoat_gloss);
        const float Dcc = GTR1(NdotH, alpha_cc);
        const float Gcc = smithG_GGX(NdotV, 0.25f);
        Vec3f f_clearcoat = Vec3f(0.25f * disney->clearcoat * (Fcc * Dcc * Gcc));

        // Integrate all terms 
        const Vec3f out = (1.0f - disney->metallic) * (lerp(f_diffuse, f_subsurface, disney->subsurface) + f_sheen) + f_specular + f_clearcoat;

        return out * clamp(NdotL, 0.0f, 1.0f);
    }

    INLINE DEVICE Vec3f pgGetDisneyBRDFSpectrum(
        const Disney::Data* disney,
        const Vec3f& wo, const Vec3f& wi,
        Shading& shading,
        const SampledSpectrum& base, const float lambda)
    {
        const Vec3f V = wo;
        const Vec3f L = wi;
        const Vec3f N = shading.n;
        const Vec3f H = (V + L) / 2.0f;
        const float NdotV = dot(N, V);
        const float NdotL = dot(N, L);
        const float NdotH = dot(N, H);
        const float LdotH /* = VdotH */ = dot(L, H);

        if (NdotV <= 0.0f || NdotL <= 0.0f)
            return 0.0f;

        const float value = base.getSpectrumFromWavelength(lambda);

        // Diffuse
        const float Fd90 = 0.5f + 2.0f * disney->roughness * LdotH * LdotH;
        const float FVd90 = fresnelSchlickT(NdotV, Fd90);
        const float FLd90 = fresnelSchlickT(NdotL, Fd90);
        float f_diffuse = value * math::inv_pi * FVd90 * FLd90;

        // Subsurface
        const float Fss90 = disney->roughness * LdotH * LdotH;
        const float FVss90 = fresnelSchlickT(NdotV, Fss90);
        const float FLss90 = fresnelSchlickT(NdotL, Fss90);
        float f_subsurface = value * math::inv_pi * 1.25f * (FVss90 * FLss90 * ((1.0f / (NdotV * NdotL)) - 0.5f) + 0.5f);

        // Sheen
        const float lumi = base.y() / constants::CIE_Y_integral;
        const float rho_tint = value / lumi;
        const float rho_sheen = lerp(1.0f, rho_tint, disney->sheen_tint);
        float f_sheen = disney->sheen * rho_sheen * pow5(1.0f - LdotH);

        // Spcular
        const Vec3f X = shading.dpdu;
        const Vec3f Y = shading.dpdv;
        const float alpha = fmaxf(0.001f, disney->roughness);
        const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
        const float ax = fmaxf(0.001f, pow2(alpha) / aspect);
        const float ay = fmaxf(0.001f, pow2(alpha) * aspect);
        const float rho_specular = lerp(1.0f, rho_tint, disney->specular_tint);
        const float Fs0 = lerp(0.08f * disney->specular * rho_specular, value, disney->metallic);
        const float FHs0 = fresnelSchlickR(LdotH, Fs0);
        const float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
        float Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
        Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);
        float f_specular = FHs0 * Ds * Gs;

        // Clearcoat
        const float Fcc = fresnelSchlickR(LdotH, 0.04f);
        const float alpha_cc = lerp(0.1f, 0.001f, disney->clearcoat_gloss);
        const float Dcc = GTR1(NdotH, alpha_cc);
        const float Gcc = smithG_GGX(NdotV, 0.25f);
        float f_clearcoat = 0.25f * disney->clearcoat * Fcc * Dcc * Gcc;

        // Integrate all terms 
        const float out = (1.0f - disney->metallic) * (lerp(f_diffuse, f_subsurface, disney->subsurface) + f_sheen) + f_specular + f_clearcoat;
        return out * clamp(NdotL, 0.0f, 1.0f);
    }

    INLINE DEVICE float pgGetDisneyPDF(
        const Disney::Data* disney,
        const Vec3f& wo, const Vec3f& wi,
        const Shading& shading
    )
    {
        const Vec3f V = -wo;
        const Vec3f L = wi;
        const Vec3f N = shading.n;

        const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
        const float specular_ratio = 1.0f - diffuse_ratio;

        const float NdotL = dot(N, L);
        const float NdotV = dot(N, V);

        if (NdotL <= 0.0f || NdotV <= 0.0f)
            return 0.0f;

        const Vec3f X = shading.dpdu;
        const Vec3f Y = shading.dpdv;
        const float alpha = fmaxf(0.001f, disney->roughness);
        const float alpha_cc = lerp(0.1f, 0.001f, disney->clearcoat_gloss);
        const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
        const float ax = fmaxf(0.001f, pow2(alpha) / aspect);
        const float ay = fmaxf(0.001f, pow2(alpha) * aspect);
        const Vec3f H = (V + L) / 2.0f;
        const float NdotH = dot(N, H);

        const float half_jacobian = 1.0f / (4.0f * dot(L, H));

        const float pdf_Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay) * NdotH * half_jacobian;
        const float pdf_Dcc = GTR1(NdotH, alpha_cc) * NdotH * half_jacobian;
        const float ratio = 1.0f / (1.0f + disney->clearcoat);
        const float pdf_specular = (pdf_Dcc + ratio * (pdf_Ds - pdf_Dcc));
        const float pdf_diffuse = NdotL * math::inv_pi;

        return fmaxf(0.0f, diffuse_ratio * pdf_diffuse + specular_ratio * pdf_specular);
    }

} // namespace prayground