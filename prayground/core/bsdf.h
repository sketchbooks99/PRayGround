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

#include <prayground/math/vec_math.h>
#include <prayground/math/util.h>
#include <prayground/math/random.h>
#include <prayground/core/util.h>
#include <prayground/optix/macros.h>

namespace prayground {

HOSTDEVICE INLINE float3 randomSampleToSphere(unsigned int& seed, const float radius, const float distance_squared)
{
    const float r1 = rnd(seed);
    const float r2 = rnd(seed);
    const float z = 1.0f + r2 * (sqrtf(1.0f - radius * radius / distance_squared) - 1.0f);

    const float phi = 2.0f * math::pi * r1;
    const float x = cosf(phi) * sqrtf(1.0f - z * z);
    const float y = sinf(phi) * sqrtf(1.0f - z * z);
    return make_float3(x, y, z);
}

HOSTDEVICE INLINE float3 randomSampleInUnitDisk(unsigned int& seed)
{
    const float theta = rnd(seed) * math::two_pi;
    const float r = rnd(seed);
    return make_float3(r * cos(theta), r * sin(theta), 0);
}

HOSTDEVICE INLINE float3 randomSampleHemisphere(unsigned int& seed)
{
    float a = rnd(seed) * 2.0f * math::pi;
    float z = sqrtf(rnd(seed));
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    return make_float3(r * cosf(a), r * sinf(a), z);
}

HOSTDEVICE INLINE float3 cosineSampleHemisphere(const float u1, const float u2)
{
    const float r = sqrtf(u2);
    const float phi = math::two_pi * u1;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(1.0f - u2);
    return make_float3(x, y, z);
}

HOSTDEVICE INLINE float3 sampleGGX(const float u1, const float u2, const float roughness)
{
    float3 p;
    const float a = fmaxf(0.001f, roughness);
    const float phi = 2.0f * math::pi * u1;
    const float cos_theta = sqrtf((1.0f - u2) / (1.0f + (a*a - 1.0f) * u2));
    const float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    p.x = cosf(phi) * sin_theta;
    p.y = sinf(phi) * sin_theta;
    p.z = cos_theta;
    return p;
}

/// @ref: https://jcgt.org/published/0007/04/01/
HOSTDEVICE INLINE float3 sampleGGXAniso(const float3& v, const float ax, const float ay, const float u0, const float u1)
{
    const float3 Vh = normalize(make_float3(ax * v.x, ay * v.y, v.z));
    const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    const float3 T1 = lensq > 0 ? make_float3(-Vh.y, Vh.x, 0) * inverseSqrt(lensq) : make_float3(1.0f, 0.0f, 0.0);
    const float3 T2 = cross(Vh, T1);
    const float r = sqrt(u0);
    const float phi = math::two_pi * u1;
    float t1 = r * cosf(phi); 
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(1.0f - t1 * t1) + s * t2;
    const float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
    const float3 Ne = normalize(make_float3(ax * Nh.x, ay * Nh.y, fmaxf(0.0f, Nh.z)));
    return Ne;
}

HOSTDEVICE INLINE float3 sampleGTR1(const float u1, const float u2, const float roughness)
{
    float3 p;
    const float a = roughness * roughness;
    const float phi = 2.0f * math::pi * u1;
    const float cos_theta = 1.0f;
}

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
    return 1.0f / ( math::pi * ax * ay * math::sqr( math::sqr(HdotX / ax) + math::sqr(HdotY / ay) + NdotH * NdotH) );
}

HOSTDEVICE INLINE float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG*alphaG;
    float b = NdotV*NdotV;
    return 1 / (NdotV + sqrt(a + b - a*b));
}

HOSTDEVICE INLINE float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 1 / (NdotV + sqrt(math::sqr(VdotX * ax) + math::sqr(VdotY * ay) + math::sqr(NdotV)));
}

HOSTDEVICE INLINE float3 refract(const float3& v, const float3& n, float ior) {
    float3 nv = normalize(v);
    float cos_i = dot(-nv, n);
    
    float3 r_out_perp = ior * (nv + cos_i*n);
    /// \note dot(v,v) = |v*v|*cos(0) = |v^2|
    float3 r_out_parallel = -sqrt(fabs(1.0f - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

/** \ref: https://knzw.tech/raytracing/?page_id=478 */
HOSTDEVICE INLINE float3 refract(const float3& wi, const float3& n, float cos_i, float ni, float nt) {
    float nt_ni = nt / ni;
    float ni_nt = ni / nt;
    float D = sqrtf(nt_ni*nt_ni - (1.0f-cos_i*cos_i)) - cos_i;
    return ni_nt * (wi - D * n);
}

}