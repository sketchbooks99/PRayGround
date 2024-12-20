#pragma once

#include <optix.h>
#include <prayground/math/vec.h>
#include <prayground/core/spectrum.h>
#include <prayground/core/keypoint.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>
#include <prayground/texture/bitmap.h>
#include <prayground/optix/cuda/device_util.cuh>

namespace prayground {

//#ifdef __CUDACC__

    /**!
     Texture should be return evaluated color
     at UV coordinates specified by intersection and/or closest-hit programs

     @param
     * - coord
       - |float2|
       - UV coordinates to evaluate color of textures.
    */

    /**! MEMO:
     There is no need to access SBT data through HitGroupData.
     It is OK to connect programs and Texture by SBT.
    */

    // Compute texture derivatives in texture space from texture derivatives in world space
    // and  ray differentials.
    inline __device__ void computeTextureDerivatives(
        Vec2f& dpdx,  // texture derivative in x (out)
        Vec2f& dpdy,  // texture derivative in y (out)
        const Vec3f& dPds,  // world space texture derivative
        const Vec3f& dPdt,  // world space texture derivative
        Vec3f rdx,   // ray differential in x
        Vec3f rdy,   // ray differential in y
        const Vec3f& normal,
        const Vec3f& rayDir)
    {
        // Compute scale factor to project differentials onto surface plane
        float s = dot(rayDir, normal);

        // Clamp s to keep ray differentials from blowing up at grazing angles. Prevents overblurring.
        const float sclamp = 0.1f;
        if (s >= 0.0f && s < sclamp)
            s = sclamp;
        if (s < 0.0f && s > -sclamp)
            s = -sclamp;

        // Project the ray differentials to the surface plane.
        float tx = dot(rdx, normal) / s;
        float ty = dot(rdy, normal) / s;
        rdx -= tx * rayDir;
        rdy -= ty * rayDir;

        // Compute the texture derivatives in texture space. These are calculated as the
        // dot products of the projected ray differentials with the texture derivatives. 
        dpdx = Vec2f(dot(dPds, rdx), dot(dPdt, rdx));
        dpdy = Vec2f(dot(dPds, rdy), dot(dPdt, rdy));
    }

    template <typename ReturnSpectrumT>
    DEVICE INLINE ReturnSpectrumT pgGetConstantTextureValue(const Vec2f& uv, void* tex_data)
    {
        const typename ConstantTexture_<ReturnSpectrumT>::Data* constant = reinterpret_cast<typename ConstantTexture_<ReturnSpectrumT>::Data*>(tex_data);
        return constant->color;
    }

    template <typename ReturnSpectrumT>
    DEVICE INLINE ReturnSpectrumT pgGetCheckerTextureValue(const Vec2f& uv, void* tex_data)
    {
        const typename CheckerTexture_<ReturnSpectrumT>::Data* checker = reinterpret_cast<typename CheckerTexture_<ReturnSpectrumT>::Data*>(tex_data);
        const bool is_odd = sinf(uv.x() * math::pi * checker->scale) * sinf(uv.y() * math::pi * checker->scale) < 0;
        return is_odd ? checker->color1 : checker->color2;
    }
    
    template <typename ReturnSpectrumT>
    DEVICE INLINE ReturnSpectrumT pgGetBitmapTextureValue(const Vec2f& uv, void* tex_data)
    {
        const BitmapTexture::Data* bitmap = reinterpret_cast<BitmapTexture::Data*>(tex_data);
        const float4 c = tex2D<float4>(bitmap->texture, uv.x(), uv.y());
        return ReturnSpectrumT(c);
    }

    template <>
    DEVICE INLINE SampledSpectrum pgGetBitmapTextureValue<SampledSpectrum>(const Vec2f& uv, void* tex_data)
    {
        const BitmapTexture::Data* bitmap = reinterpret_cast<BitmapTexture::Data*>(tex_data);
        const float4 c = tex2D<float4>(bitmap->texture, uv.x(), uv.y());
        return rgb2spectrum(Vec3f(c));
    }

    template <typename ReturnSpectrumT>
    DEVICE INLINE ReturnSpectrumT pgGetGradientTextureValue(const Vec2f& uv, void* tex_data)
    {
        const typename GradientTexture_<ReturnSpectrumT>::Data* gradient = reinterpret_cast<typename GradientTexture_<ReturnSpectrumT>::Data*>(tex_data);
        // Calculate t value in [0, 1] range from uv coordinates and gradient direction
        const float t = dot(uv, normalize(gradient->dir));
        // Find two key points that sandwich t
        float min_t = 0.0f;
        float max_t = 1.0f;
        for (int i = 0; i < gradient->num_keypoints - 1; i++) {
            min_t = fminf(min_t, gradient->keypoints[i].t);
            max_t = fmaxf(max_t, gradient->keypoints[i].t);
            if (t >= gradient->keypoints[i].t && t < gradient->keypoints[i + 1].t) {
                return Keypoint<ReturnSpectrumT>::ease(gradient->keypoints[i], gradient->keypoints[i + 1], t, gradient->ease_type);
            }
        }
        min_t = fminf(min_t, gradient->keypoints[gradient->num_keypoints - 1].t);
        max_t = fmaxf(max_t, gradient->keypoints[gradient->num_keypoints - 1].t);

        // If t is out of range, return the color of the nearest keypoint
        if (t < min_t) {
            return gradient->keypoints[0].value;
        } else if (t > max_t) {
            return gradient->keypoints[gradient->num_keypoints - 1].value;
        }
        return ReturnSpectrumT();
    }

//#endif

} // namespace prayground
