#pragma once

#include <cuda/random.h>
#include <prayground/core/bsdf.h>
#include <prayground/core/onb.h>
#include <prayground/core/color.h>
#include <prayground/core/interaction.h>
#include <prayground/material/disney.h>

CALLABLE_FUNC void DC_FUNC(sample_disney)(SurfaceInteraction* si, void* mat_data)
{
    const DisneyData* disney = reinterpret_cast<DisneyData*>(mat_data);

    if (disney->twosided)
        si->n = faceforward(si->n, -si->wi, si->n);

    unsigned int seed = si->seed;
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    // const float diffuse_ratio = 1.0f - disney->metallic;
    Onb onb(si->n);

    if (rnd(seed) < diffuse_ratio)
    {
        float3 w_in = cosineSampleHemisphere(z1, z2);
        onb.inverseTransform(w_in);
        si->wo = normalize(w_in);
    }
    else
    {
        /// @todo Change sampling functions according to the ratio to choose specular or clearcoat pdf
        /// ratio = 1.0f / (1.0f + clearcoat);

        float3 h = sampleGGX(z1, z2, disney->roughness);
        onb.inverseTransform(h);
        si->wo = normalize(reflect(si->wi, h));
    }
    si->seed = seed;
    si->radiance_evaled = false;
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
CALLABLE_FUNC float3 CC_FUNC(bsdf_disney)(SurfaceInteraction* si, void* mat_data)
{   
    const DisneyData* disney = reinterpret_cast<DisneyData*>(mat_data);
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
        disney->base_program_id, si, disney->base_tex_data
    );

    // Diffuse term (diffuse, subsurface, sheen) ======================
    // Diffuse
    const float Fd90 = 0.5f + 2.0f * disney->roughness * LdotH*LdotH;
    const float FVd90 = fresnelSchlickT(NdotV, Fd90);
    const float FLd90 = fresnelSchlickT(NdotL, Fd90);
    const float3 f_diffuse = (base_color / math::pi) * FVd90 * FLd90;

    // Subsurface
    const float Fss90 = disney->roughness * LdotH*LdotH;
    const float FVss90 = fresnelSchlickT(NdotV, Fss90);
    const float FLss90 = fresnelSchlickT(NdotL, Fss90); 
    const float3 f_subsurface = (base_color / math::pi) * 1.25f * (FVss90 * FLss90 * ((1.0f / (NdotV * NdotL)) - 0.5f) + 0.5f);

    // Sheen
    const float3 rho_tint = base_color / luminance(base_color);
    const float3 rho_sheen = lerp(make_float3(1.0f), rho_tint, disney->sheen_tint); 
    const float3 f_sheen = disney->sheen * rho_sheen * powf(1.0f - LdotH, 5.0f);

    // Specular term (specular, clearcoat) ============================
    // Spcular
    const float3 rho_specular = lerp(make_float3(1.0f), rho_tint, disney->specular_tint);
    const float3 Fs0 = lerp(0.08f * disney->specular * rho_specular, base_color, disney->metallic);
    const float3 FHs0 = fresnelSchlickR(LdotH, Fs0);
    const float alpha = fmaxf(0.001f, disney->roughness * disney->roughness); // Remapping of roughness
    const float Ds = GTR2(NdotH, alpha);
    const float alpha_g = powf(0.5f*disney->roughness + 0.5f, 2.0f);
    const float Gs = geometrySmith(N, V, L, alpha_g);
    const float3 f_specular = FHs0 * Ds * Gs / (4.0f * NdotV * NdotL);

    // Clearcoat
    const float Fcc = fresnelSchlickR(LdotH, 0.04f);
    const float alpha_cc = 0.1f + (0.001f - 0.1f) * disney->clearcoat_gloss; // lerp
    const float Dcc = GTR1(NdotH, alpha_cc);
    const float Gcc = geometrySmith(N, V, L, 0.25f);
    const float3 f_clearcoat = make_float3( 0.25f * disney->clearcoat * (Fcc * Dcc * Gcc) / (4.0f * NdotV * NdotL) ); 

    const float3 out = ( 1.0f - disney->metallic ) * ( lerp( f_diffuse, f_subsurface, disney->subsurface ) + f_sheen ) + f_specular + f_clearcoat;
    return out * NdotL;
}

/**
 * @ref (temporal) http://simon-kallweit.me/rendercompo2015/report/#adaptivesampling
 * 
 * @todo Investigate correct evaluation of PDF.
 */
CALLABLE_FUNC float DC_FUNC(pdf_disney)(SurfaceInteraction* si, void* mat_data)
{
    const DisneyData* disney = reinterpret_cast<DisneyData*>(mat_data);

    const float3 V = -si->wi;
    const float3 L = si->wo;
    const float3 N = si->n;

    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    const float specular_ratio = 1.0f - diffuse_ratio;

    const float NdotL = abs(dot(N, L));
    const float NdotV = abs(dot(N, V));

    const float alpha = fmaxf(0.001f, disney->roughness * disney->roughness);
    const float alpha_cc = 0.1f + (0.001f - 0.1f) * disney->clearcoat_gloss; // lerp
    const float3 H = normalize(V + L);
    const float NdotH = abs(dot(H, N));

    const float pdf_Ds = GTR2(NdotH, alpha);
    const float pdf_Dcc = GTR1(NdotH, alpha_cc);
    const float ratio = 1.0f / (1.0f + disney->clearcoat);
    const float pdf_specular = (pdf_Dcc + ratio * (pdf_Ds - pdf_Dcc)) / (4.0f * NdotH);
    const float pdf_diffuse = NdotL / math::pi;

    return diffuse_ratio * pdf_diffuse + specular_ratio * pdf_specular;
}