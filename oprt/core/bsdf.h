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

#include <cuda/random.h>
#include <sutil/vec_math.h>
#include "../core/util.h"
#include "../optix/util.h"

namespace oprt {

HOSTDEVICE INLINE float3 randomSampleHemisphere(unsigned int& seed) {
    float a = rnd(seed) * 2.0f * M_PIf;
    float z = sqrtf(rnd(seed));
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    return make_float3(r * cosf(a), r * sinf(a), z);
}

HOSTDEVICE INLINE float3 cosineSampleHemisphere(const float u1, const float u2)
{
    float3 p;
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
    return p;
}

HOSTDEVICE INLINE float3 sampleGGX(const float u1, const float u2, const float roughness)
{
    float3 p;
    const float a = roughness * roughness;
    const float phi = 2.0f * M_PIf * u1;
    const float cos_theta = sqrtf((1.0f - u2) / (1.0f + (a*a - 1.0f) * u2));
    const float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    p.x = cosf(phi) * sin_theta;
    p.y = sinf(phi) * sin_theta;
    p.z = cos_theta;
    return p;
}

HOSTDEVICE INLINE float3 sampleGTR1(const float u1, const float u2, const float roughness)
{
    float3 p;
    const float a = roughness * roughness;
    const float phi = 2.0f * M_PIf * u1;
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
 * @note Difference from original ... PI -> M_PIf
 * @param a : roughness of the surface. [0,1]
 */
HOSTDEVICE INLINE float GTR1(float NdotH, float a)
{
    if (a >= 1) return 1/M_PIf;
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return (a2-1) / (M_PIf*log(a2)*t);
}

/** 
 * @ref: https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
 * @brief Trowbridge-Reitz GGX function for NDF, GTR(gamma = 2)
 * @note Difference from original ... PI -> M_PIf
 * @param a : roughness of the surface. [0,1]
 **/ 
HOSTDEVICE INLINE float GTR2(float NdotH, float a)
{
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return a2 / (M_PIf * t*t);
}

/**
 * @brief Geometry function of combining GGX and Schlick-Beckmann approximation
 * 
 * @param n : normal
 * @param v : view vector
 * @param k : remapping of roughness that depends on lighting context (direct or IBL).
 */
HOSTDEVICE INLINE float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0f);
    float k = (r*r) / 8.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}

/**
 * @brief Geometry function that takes account view direction (obstruction) and light direction (shadowing).
 * 
 * @param n : normal
 * @param v : view vector
 * @param l : light vector
 * @param k : remapping of roughness that depends on lighting context (direct or IBL).
 */
HOSTDEVICE INLINE float geometrySmith(const float3& n, const float3& v, const float3& l, float roughness)
{
    const float NdotV = fmaxf(dot(n, v), 0.0f);
    const float NdotL = fmaxf(dot(n, l), 0.0f);
    const float ggxV = geometrySchlickGGX(NdotV, roughness);
    const float ggxL = geometrySchlickGGX(NdotL, roughness);
    return ggxV * ggxL;
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