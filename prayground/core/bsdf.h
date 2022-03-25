// Copyright Disney Enterprises, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License
// and the following modification to it: Section 6 Trademarks.
// deleted and replaced with:
//
// 6. Trademarks. This License does not grant permission to use the
// trade names, trademarks, service marks, or product names of the
// Licensor and its affiliates, except as required for reproducing
// the content of the NOTICE file.
//
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <prayground/math/vec.h>
#include <prayground/math/util.h>
#include <prayground/math/random.h>
#include <prayground/core/util.h>
#include <prayground/optix/macros.h>

namespace prayground {

HOSTDEVICE INLINE Vec3f randomSampleToSphere(unsigned int& seed, const float radius, const float distance_squared)
{
    const float r1 = rnd(seed);
    const float r2 = rnd(seed);
    const float z = 1.0f + r2 * (sqrtf(1.0f - radius * radius / distance_squared) - 1.0f);

    const float phi = 2.0f * math::pi * r1;
    const float x = cosf(phi) * sqrtf(1.0f - z * z);
    const float y = sinf(phi) * sqrtf(1.0f - z * z);
    return Vec3f(x, y, z);
}

HOSTDEVICE INLINE Vec3f randomSampleInUnitDisk(unsigned int& seed)
{
    const float theta = rnd(seed) * math::two_pi;
    const float r = rnd(seed);
    return Vec3f(r * cos(theta), r * sin(theta), 0);
}

HOSTDEVICE INLINE Vec3f randomSampleHemisphere(unsigned int& seed)
{
    float phi = math::two_pi * rnd(seed);
    float z = sqrtf(rnd(seed));
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    return Vec3f(r * cosf(phi), r * sinf(phi), z);
}

HOSTDEVICE INLINE Vec3f cosineSampleHemisphere(const float u1, const float u2)
{
    const float r = sqrtf(u1);
    const float phi = math::two_pi * u2;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - x * x - y * y));
    return Vec3f(x, y, z);
}

HOSTDEVICE INLINE Vec3f sampleGTR1(const float u1, const float u2, const float roughness)
{
    const float a = fmaxf(0.001f, roughness);
    const float a2 = a * a;
    const float cos_theta = sqrtf(fmaxf(0.0f, (1.0f - powf(a2, 1.0f - u1) / (1.0f - a2))));
    const float sin_theta = sqrtf(1.0f - pow2(cos_theta));
    const float phi = math::two_pi * u2;

    return Vec3f(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);
}

HOSTDEVICE INLINE Vec3f sampleGGX(const float u1, const float u2, const float roughness)
{
    const float a = fmaxf(0.001f, roughness);
    const float phi = math::two_pi * u1;
    const float cos_theta = sqrtf((1.0f - u2) / (1.0f + (a*a - 1.0f) * u2));
    const float sin_theta = sqrtf(1.0f - pow2(cos_theta));

    return Vec3f(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);
}

/// @ref: https://jcgt.org/published/0007/04/01/
HOSTDEVICE INLINE Vec3f sampleGGXAniso(const Vec3f& v, const float ax, const float ay, const float u0, const float u1)
{
    const Vec3f Vh = normalize(Vec3f(ax * v[0], ay * v[1], v[2]));
    const float lensq = Vh[0] * Vh[0] + Vh[1] * Vh[1];
    const float inv_sqrt_lensq = 1.0f / sqrtf(lensq);
    const Vec3f T1 = lensq > 0 ? Vec3f(-Vh[1], Vh[0], 0) * inv_sqrt_lensq : Vec3f(1.0f, 0.0f, 0.0);
    const Vec3f T2 = cross(Vh, T1);
    const float r = sqrt(u0);
    const float phi = math::two_pi * u1;
    float t1 = r * cosf(phi); 
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh[2]);
    t2 = (1.0f - s) * sqrtf(1.0f - t1 * t1) + s * t2;
    const Vec3f Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    const Vec3f Ne = normalize(Vec3f(ax * Nh[0], ay * Nh[1], fmaxf(0.0f, Nh[2])));
    return Ne;
}

/// @todo : Implement
// HOSTDEVICE INLINE Vec3f sampleGTR1(const float u1, const float u2, const float roughness)
// {
//     Vec3f p;
//     const float a = pow2(roughness);
//     const float phi = 2.0f * math::pi * u1;
//     const float cos_theta = 1.0f;
// }

/** 
 * @ref: http://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission.html
 * 
 * @note cos_i must be positive, and this function does not verify
 * whether the ray goes into a surface (is cos_i negative?) or not. 
 **/
HOSTDEVICE INLINE float fresnel(float cos_i, float ni, float nt) {
    const float sin_i = sqrtf(fmaxf(0.0f, 1.0f-cos_i*cos_i));
    const float sin_t = ni / nt * sin_i;
    const float cos_t = sqrtf(fmaxf(0.0f, 1.0f-sin_t*sin_t));

    const float r_parl = ((nt * cos_i) - (ni * cos_t)) / 
                   ((nt * cos_i) + (ni * cos_t));
    const float r_perp = ((ni * cos_i) - (nt * cos_t)) / 
                   ((ni * cos_i) + (nt * cos_t));

    return 0.5f * (r_parl*r_parl + r_perp*r_perp);
}

/** 
 * @brief Compute fresnel reflectance from cosine and index of refraction using Schlick approximation. 
 **/
HOSTDEVICE INLINE float fresnelSchlick(float cos_i, float ior) {
    float r0 = (1.0f-ior) / (1.0f+ior);
    r0 = r0 * r0;
    return r0 + (1.0f-r0)*powf((1.0f-cos_i),5);
}

/**
 * @ref https://rayspace.xyz/CG/contents/Disney_principled_BRDF/ 
 * 
 * @brief Schlick approximation of fresnel reflectance.
 * 
 * @param cos_i : cosine between normal and incident angle.
 * @param f0 : base reflectance (when incident angle is perpendicular with a surface)
 **/
template <class T>
HOSTDEVICE INLINE T fresnelSchlickR(float cos_i, T f0) {
    return f0 + (1.0f-f0)*powf((1.0f - cos_i), 5.0f);
}

template <class T>
HOSTDEVICE INLINE T fresnelSchlickT(float cos_i, T f90) {
    return 1.0f + (f90-1.0f)*powf((1.0f - cos_i), 5.0f);
}

/**
 * @ref Physically Based Shading at Disney, https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
 * @brief NDF from Berry (1923), GTR(gamma = 1)
 * @note Difference from original ... PI -> math::pi
 * @param a : roughness of the surface. [0,1]
 */
HOSTDEVICE INLINE float GTR1(float NdotH, float a)
{
    if (a >= 1) return 1/math::pi;
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return (a2-1) / (math::pi*log(a2)*t);
}

/** 
 * @ref: https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
 * @brief Trowbridge-Reitz GGX function for NDF, GTR(gamma = 2)
 * @note Difference from original ... PI -> math::pi
 * @param a : roughness of the surface. [0,1]
 **/ 
HOSTDEVICE INLINE float GTR2(float NdotH, float a)
{
    float a2 = a*a;
    float t = 1.0f - (1.0f-a2)*NdotH*NdotH;
    return a2 / (math::pi * t*t);
}

HOSTDEVICE INLINE float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
    return 1.0f / ( math::pi * ax * ay * pow2( pow2(HdotX / ax) + pow2(HdotY / ay) + NdotH * NdotH) );
}

HOSTDEVICE INLINE float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG*alphaG;
    float b = NdotV*NdotV;
    return 1 / (NdotV + sqrt(a + b - a*b));
}

HOSTDEVICE INLINE float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 1 / (NdotV + sqrt(pow2(VdotX * ax) + pow2(VdotY * ay) + pow2(NdotV)));
}

HOSTDEVICE INLINE Vec3f reflect(const Vec3f& wo, const Vec3f& n)
{
    return wo - 2.0f * dot(wo, n) * n;
}

HOSTDEVICE INLINE Vec3f refract(const Vec3f& wo, const Vec3f& n, float ior) {
    float cos_theta = dot(-normalize(wo), n);
    
    Vec3f r_out_perp = ior * (wo + cos_theta*n);
    Vec3f r_out_parallel = -sqrt(fabs(1.0f - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

/** \ref: https://knzw.tech/raytracing/?page_id=478 */
HOSTDEVICE INLINE Vec3f refract(const Vec3f& wo, const Vec3f& n, float cos_theta, float ni, float nt) {
    float nt_ni = nt / ni;
    float ni_nt = ni / nt;
    float D = sqrtf(nt_ni*nt_ni - (1.0f-cos_theta*cos_theta)) - cos_theta;
    return ni_nt * (wo - D * n);
}

/* Phase function by Henyey and Greenstein */
HOSTDEVICE INLINE float phaseHenyeyGreenstein(float cos_theta, float g)
{
    const float denom = 1.0f + g * g + 2.0f * g * cos_theta;
    return (1.0f / (4.0f * math::pi)) * (1.0f - g * g) / (denom * sqrtf(denom));
}

HOSTDEVICE INLINE Vec3f sampleHenyeyGreenstein(const Vec2f& u, float g)
{
    float cos_theta = 0.0f;
    if (fabs(g) < math::eps)
        cos_theta = 1.0f - 2.0 * u[0];
    else
    {
        float sqr_term = (1.0f - g * g) / (1.0f + g - 2.0f * g * u[0]);
        cos_theta = -(1.0f + g * g - pow2(sqr_term)) / (2.0f * g);
    }

    // Compute wi
    const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - pow2(cos_theta)));
    const float phi = 2 * math::pi * u[1];
    return Vec3f(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);
}

}