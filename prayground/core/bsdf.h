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
#include <prayground/math/matrix.h>
#include <prayground/math/util.h>
#include <prayground/math/random.h>
#include <prayground/core/spectrum.h>
#include <prayground/core/util.h>
#include <prayground/optix/macros.h>
#include <prayground/core/sampler.h>


namespace prayground {

    HOSTDEVICE INLINE Vec3f randomSampleToSphere(uint32_t& seed, const float radius, const float distance_squared)
    {
        Vec2f u = UniformSampler::get2D(seed);
        const float z = 1.0f + u[1] * (sqrtf(1.0f - radius * radius / distance_squared) - 1.0f);

        const float phi = 2.0f * math::pi * u[0];
        const float x = cosf(phi) * sqrtf(1.0f - z * z);
        const float y = sinf(phi) * sqrtf(1.0f - z * z);
        return Vec3f(x, y, z);
    }

    HOSTDEVICE INLINE Vec3f randomSampleOnUnitSphere(uint32_t& seed)
    {
        const float theta = 2.0f * math::pi * rnd(seed);
        const float phi = 2.0f * math::pi * acosf(1.0f - 2.0f * rnd(seed));
        const float x = sinf(phi) * cosf(theta);
        const float y = sinf(phi) * sinf(theta);
        const float z = cosf(theta);

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
        float r = sqrtf(1.0f - z * z);
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

    HOSTDEVICE INLINE Vec3f sampleGGXAniso(const float ax, const float ay, const float u0, const float u1)
    {
        float phi = atanf(tanf(2.0f * math::pi * u0) * ay / ax);
        if (u0 >= 0.75f)
            phi += math::two_pi;
        else if (u0 > 0.25f)
            phi += math::pi;

        const float cosPhi = cosf(phi);
        const float sinPhi = sinf(phi);
        const float A = pow2(cosPhi / ax) + pow2(sinPhi / ay);
        const float theta = atanf(sqrtf(u1 / ((1.0f - u1) * A)));

        return Vec3f(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
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
        if (a >= 1.0f) return math::inv_pi;
        float a2 = a*a;
        return (a2 - 1.0f) / (math::pi * logf(a2) * (1.0f + (a2 - 1.0f) * NdotH * NdotH));
    }

    /** 
     * @ref: https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
     * @brief Trowbridge-Reitz GGX function for NDF, GTR(gamma = 2)
     * @note Difference from original ... PI -> math::pi
     * @param a : roughness of the surface. [0,1]
     **/ 
    HOSTDEVICE INLINE float GTR2(float NdotH, float a)
    {
        float a2 = a * a;
        float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
        return a2 / (math::pi * t * t);
    }

    HOSTDEVICE INLINE float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
    {
        return 1.0f / (math::pi * ax * ay * pow2(pow2(HdotX / ax) + pow2(HdotY / ay) + pow2(NdotH)));
    }

    HOSTDEVICE INLINE float smithG_GGX(float NdotV, float alphaG)
    {
        float a = alphaG*alphaG;
        float b = NdotV*NdotV;
        return 1.0f / (NdotV + sqrtf(a + b - a*b));
    }

    HOSTDEVICE INLINE float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
    {
        return 1.0f / (NdotV + sqrtf(pow2(VdotX * ax) + pow2(VdotY * ay) + pow2(NdotV)));
    }

    HOSTDEVICE INLINE Vec3f reflect(const Vec3f& wo, const Vec3f& n)
    {
        return wo - 2.0f * dot(wo, n) * n;
    }

    HOSTDEVICE INLINE Vec3f refract(const Vec3f& wo, const Vec3f& n, float ior) {
        float cos_theta = dot(-normalize(wo), n);
    
        Vec3f r_out_perp = ior * (wo + cos_theta*n);
        Vec3f r_out_parallel = -sqrtf(fabs(1.0f - dot(r_out_perp, r_out_perp))) * n;
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

    HOSTDEVICE INLINE Vec3f evalSensitivity(float opd, const Vec3f& shift) 
    {
        // Use Gaussian fits, given by 3 parameters: val, pos and var
        float phase = math::two_pi * opd;
        Vec3f val = Vec3f(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
        Vec3f pos = Vec3f(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
        Vec3f var = Vec3f(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);
        Vec3f xyz = val * sqrt(var * math::two_pi) 
                        * Vec3f(cosf(pos.x() * phase + shift.x()), cosf(pos.y() * phase + shift.y()), cosf(pos.z() * phase + shift.z()))
                        * Vec3f(expf(-var.x() * pow2(phase)), expf(-var.y() * pow2(phase)), expf(-var.z() * pow2(phase)));
        xyz.x() += 9.7470e-14f * sqrtf(math::two_pi * 4.5282e+09f) * cosf(2.2399e+06f * phase + shift[0]) * exp(-4.5282e+09f * phase * phase);
        return xyz / 1.0685e-7f;
    }

    HOSTDEVICE INLINE Vec2f fresnelDielectricPolarized(float cos_theta, float ior) {
        float cos_theta2 = pow2(clamp(cos_theta, 0.0f, 1.0f));
        float sin_theta2 = 1.0f - cos_theta2;

        float t0 = max(ior * ior - sin_theta2, 0.0f);
        float t1 = t0 + cos_theta2;
        float t2 = 2.0f * sqrtf(t0) * cos_theta;
        float Rs = (t1 - t2) / (t1 + t2);

        float t3 = cos_theta2 * t0 + sin_theta2 * sin_theta2;
        float t4 = t2 * sin_theta2;
        float Rp = Rs * (t3 - t4) / (t3 + t4);

        return Vec2f(Rp, Rs);
    }

    HOSTDEVICE INLINE void fresnelConductorPolarized(float cos_theta, const Vec3f& n, const Vec3f& k, Vec3f& Rp, Vec3f& Rs)
    {
        float cos_theta2 = pow2(clamp(cos_theta, 0.0f, 1.0f));
        float sin_theta2 = 1.0f - cos_theta2;
        Vec3f n2 = n * n;
        Vec3f k2 = k * k;

        Vec3f t0 = n2 - k2 - sin_theta2;
        Vec3f a2plusb2 = sqrt(t0 * t0 + 4.0f * n2 * k2);
        Vec3f t1 = a2plusb2 + cos_theta2;
        Vec3f a = sqrt(max(0.5f * (a2plusb2 + t0), 0.0f));
        Vec3f t2 = 2.0f * a * cos_theta;
        Rs = (t1 - t2) / (t1 + t2);

        Vec3f t3 = cos_theta2 * a2plusb2 + pow2(sin_theta2);
        Vec3f t4 = t2 * sin_theta2;
        Rp = Rs * (t3 - t4) / (t3 + t4);
    }

    HOSTDEVICE INLINE void fresnelConductorPhasePolarized(float cos_theta, float eta1, const Vec3f& eta2, const Vec3f& kappa2, Vec3f& phi_p, Vec3f& phi_s) 
    {
        Vec3f k2 = kappa2 / eta2;
        Vec3f sin_theta_sqr = 1.0f - pow2(cos_theta);
        Vec3f A = eta2 * eta2 * (1.0f - k2 * k2) - eta1 * eta1 * sin_theta_sqr;
        Vec3f B = sqrt(A * A + pow2(2.0f * eta2 * eta2 * k2));
        Vec3f U = sqrt((A + B) / 2.0f);
        Vec3f V = max(sqrt((B - A) / 2.0f), 0.0f);

        Vec3f s_a = 2.0f * eta1 * V * cos_theta;
        Vec3f s_b = U * U + V * V - pow2(eta1 * cos_theta);
        phi_s = Vec3f(atan2(s_a.x(), s_b.x()), atan2(s_a.y(), s_b.y()), atan2(s_a.z(), s_b.z()));

        Vec3f p_a = 2.0f * eta1 * eta2 * eta2 * cos_theta * (2.0f * k2 * U - (1.0f - k2 * k2) * V);
        Vec3f p_b = pow2(eta2 * eta2 * (1.0 + k2 * k2) * cos_theta) - eta1 * eta1 * (U * U + V * V);
        phi_p = Vec3f(atan2(p_a.x(), p_b.x()), atan2(p_a.y(), p_b.y()), atan2(p_a.z(), p_b.z()));
    }

   

    HOSTDEVICE INLINE Vec3f fresnelAiry(
        float eta1, // incident medium
        float cos_theta,
        const Vec3f& ior,
        const Vec3f& extinction,
        float tf_thickness,
        float tf_ior
    )
    {
        float eta2 = max(eta1, tf_ior);
        Vec3f eta3 = ior;

        Vec3f kappa3 = extinction;
        float cos_theta2 = pow2(cos_theta);
        float cos_thetaT = sqrtf(1.0f - (1.0f - cos_theta2) * pow2(eta1 / eta2));

        // First interface
        Vec2f R12 = fresnelDielectricPolarized(cos_theta, eta2 / eta1);
        if (cos_thetaT <= 0.0f) {
            // Total internal reflection
            R12 = 1.0f;
        }
        Vec2f T121 = 1.0f - R12;

        // Second interface
        Vec3f R23p, R23s;
        fresnelConductorPolarized(cos_theta2, eta3 / eta2, kappa3 / eta2, R23p, R23s);

        // Phase shift
        float cos_b = cos(atanf(eta2 / eta1));
        Vec2f phi21 = Vec2f(cos_theta < cos_b ? 0.0f : math::pi, math::pi);
        Vec3f phi23p, phi23s;
        fresnelConductorPhasePolarized(cos_thetaT, eta2, eta3, kappa3, phi23p, phi23s);

        Vec3f r123p = max(sqrt(R12.x() * R23p), 0.0f);
        Vec3f r123s = max(sqrt(R12.y() * R23s), 0.0f);

        // Iridescence term
        Vec3f I = Vec3f(0.0f);
        Vec3f Cm, Sm;

        // Optical path difference
        float dist_meters = tf_thickness * 1e-9f;
        float opd = 2.0f * eta2 * cos_thetaT * dist_meters;

        // For parallel polarization
        // Reflectance term for m=0 (DC term amplitude)
        Vec3f Rs = (pow2(T121.x()) * R23p) / (1.0f - R12.x() * R23p);
        I += R12.x() + Rs;

        // Reflectance term for m>0 (pairs of diracs)
        Cm = Rs - T121.x();
        for (int m = 1; m <= 2; m++) {
            Cm *= r123p;
            Sm = 2.0f * evalSensitivity(float(m) * opd, float(m) * (phi23p + phi21.x()));
            I += Cm * Sm;
        }

        // For perpendicular polarization
        // Reflectance term for m=0 (DC term amplitude)
        Vec3f Rp = (pow2(T121.y()) * R23s) / (1.0f - R12.y() * R23s);
        I += R12.y() + Rp;

        // Reflectance term for m>0 (pairs of diracs)
        Cm = Rp - T121.y();
        for (int m = 1; m <= 2; m++) {
            Cm *= r123s;
            Sm = 2.0f * evalSensitivity(float(m) * opd, float(m) * (phi23s + phi21.y()));
            I += Cm * Sm;
        }

        // Average parallel and perpendicular polarizations
        I *= 0.5f;

        // Convert to RGB reflectance
        return XYZToSRGB(I);
    }

} // namespace prayground