#pragma once

#include <cuda/random.h>
#include <sutil/vec_math.h>
#include "../core/util.h"
#include "../optix/util.h"

namespace oprt {

HOSTDEVICE INLINE float3 random_sample_hemisphere(unsigned int seed) {
    // return make_float3(0.0f, 1.0f, 0.0f);
    float a = rnd(seed) * 2.0f * M_PIf;
    float z = rnd(seed);
    auto r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    return make_float3(r * cosf(a), r * sinf(a), z);
}

HOSTDEVICE INLINE float3 cosine_sample_hemisphere(const float u1, const float u2)
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

/** 
 * \ref: http://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission.html
 * 
 * \note cos_i must be positive, and this function does not verify
 * whether the ray goes into a surface (is cos_i negative?) or not. 
 **/
HOSTDEVICE INLINE float fr(float cos_i, float ni, float nt) {
    const float sin_i = sqrtf(fmaxf(0.0f, 1.0f-cos_i*cos_i));
    const float sin_t = ni / nt * sin_i;
    const float cos_t = sqrtf(fmaxf(0.0f, 1.0f-sin_t*sin_t));

    const float r_parl = ((nt * cos_i) - (ni * cos_t)) / 
                   ((nt * cos_i) + (ni * cos_t));
    const float r_perp = ((ni * cos_i) - (nt * cos_t)) / 
                   ((ni * cos_i) + (nt * cos_t));

    return 0.5f * (r_parl*r_parl + r_perp*r_perp);
}

/** Compute fresnel reflectance from cosine and index of refraction using Schlick approximation. */
HOSTDEVICE INLINE float fr(float cos_i, float ior) {
    float r0 = (1.0f-ior) / (1.0f+ior);
    r0 = r0 * r0;
    return r0 + (1.0f-r0)*powf((1.0f-cos_i),5);
}

/** \ref: https://rayspace.xyz/CG/contents/Disney_principled_BRDF/ */
HOSTDEVICE INLINE float fr_schlick(float cos_i, float f0) {
    return f0 + (1.0f-f0)*powf((1.0f-cos_i),5);
}
HOSTDEVICE INLINE float ft_schlick(float cos_i, float f90) {
    return 1.0f + (f90-1.0f)*powf((1.0f - cos_i), 5);
}

/** 
 * @ref: https://learnopengl.com/PBR/Theory
 * 
 * @brief Trowbridge-Reitz GGX function for NDF
 * 
 * @param n : normal
 * @param h : half vector
 * @param a : roughness of the surface. [0,1]
 **/ 
HOSTDEVICE INLINE float trowbridge_reitz_ggx(const float3& n, const float3& h, float a) {
    const float a2 = a*a;
    const float NdotH = fmaxf(dot(n, h), 0.0f);
    const float denom = NdotH*NdotH * (a2-1.0f)+1.0f;
    return a2 / (M_PIf * denom * denom);
}

/**
 * @brief Geometry function of combining GGX and Schlick-Beckmann approximation
 * 
 * @param n : normal
 * @param v : view vector
 * @param k : remapping of roughness that depends on lighting context (direct or IBL).
 */
HOSTDEVICE INLINE float geometry_schlick_ggx(float NdotV, float k)
{
    return NdotV / (NdotV * (1-k) + k);
}

/**
 * @brief Geometry function that takes account view direction (obstruction) and light direction (shadowing).
 * 
 * @param n : normal
 * @param v : view vector
 * @param l : light vector
 * @param k : remapping of roughness that depends on lighting context (direct or IBL).
 */
HOSTDEVICE INLINE float geometry_smith(const float3& n, const float3& v, const float3& l, float k)
{
    const float NdotV = fmaxf(dot(n, v), 0.0f);
    const float NdotL = fmaxf(dot(n, l), 0.0f);
    const float ggxV = geometry_schlick_ggx(NdotV, k);
    const float ggxL = geometry_schlick_ggx(NdotL, k);
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

HOSTDEVICE INLINE float3 retro_transmit(const float3& i, const float3& n)
{
    return -i + 2.0f * n * dot(n, i);
}

}