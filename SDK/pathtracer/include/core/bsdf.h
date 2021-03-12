#pragma once

#include <sutil/vec_math.h>
#include <core/util.h>

namespace pt {

HOSTDEVICE INLINE void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PTf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

/** 
 * \ref: http://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission.html
 * 
 * \note cos_i should be positive and checking if ray goes into 
 * dielectric surface (is cos_i negative?) is never performed in this function. 
 **/
HOSTDEVICE INLINE float fr(float cos_i, float ni, float nt) {
    float sin_i = sqrtf(fmaxf((float)0, 1-cos_i*cos_i));
    float sin_t = ni / nt * sin_i;
    float cos_t = sqrtf(fmaxf((float)0, 1-sin_t*sin_t));

    float r_parl = ((nt * cos_i) - (ni * cos_t)) / 
                   ((nt * cos_i) + (ni * cos_t));
    float r_perp = ((ni * cos_i) - (nt * cos_t)) / 
                   ((ni * cos_i) + (nt * cos_t));

    return (float)0.5 * (r_parl*r_parl + r_perp*r_perp);
}

/** Compute fresnel reflactance from cosine and index of refraction */
HOSTDEVICE INLINE float fr(float cos_i, float ior) {
    auto r0 = (1-ior) / (1+ior);
    r0 = r0 * r0;
    return r0 + (1-r0)*powf((1-cos_i),5);
}

/** \ref: https://rayspace.xyz/CG/contents/Disney_principled_BRDF/ */
HOSTDEVICE INLINE float fr_schlick(float cos_i, float f0) {
    return f0 + (1-f0)*powf((1-cos_i),5);
}
HOSTDEVICE INLINE float ft_schlick(float cos_i, float f90) {
    return 1 + (f90-1)*powf((1 - cos_i), 5);
}

/** 
 * \ref: https://qiita.com/_Pheema_/items/f1ffb2e38cc766e6e668
 * 
 * @param
 * - n : normal
 * - h : half vector
 * - rough : roughness of the surface. [0,1]
 **/ 
HOSTDEVICE INLINE float ggx(const float3& n, const float3& h, float rough) {
    float d = dot(n, h);
    float dd = d*d;
    float a = (1.0 - (1.0-rough*rough)*dd);
    float denom = pi * a*a;
    return rough*rough / denom;
}

HOSTDEVICE INLINE float3 reflect(const float3& v, const float3& n) {
    return v - 2*dot(v,n) * n;
}

HOSTDEVICE INLINE float3 refract(const float3& v, const float3& n, float ior) {
    float3 nv = normalize(v);
    float cos_i = dot(-nv, n);
    
    float3 r_out_perp = ior * (nv + cos_i*n);
    /// \note dot(v,v) = |v*v|*cos(0) = |v^2|
    float3 r_out_parallel = -sqrt(fabs(1.0 - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

/** \ref: https://knzw.tech/raytracing/?page_id=478 */
HOSTDEVICE INLINE float3 refract(const float3& wi, const float3& n, float cos_i, float ni, float nt) {
    float nt_ni = nt / ni;
    float ni_nt = ni / nt;
    float D = sqrtf(nt_ni*nt_ni - (1-cos_i*cos_i)) - cos_i;
    return ni_nt * (wi - D * n);
}

}