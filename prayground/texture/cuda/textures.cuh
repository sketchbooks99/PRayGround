#pragma once

#include <optix.h>
#include <prayground/math/vec.h>
#include <prayground/core/spectrum.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>
#include <prayground/texture/bitmap.h>

namespace prayground {

#ifdef __CUDACC__

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
    
    // Bitmap ------------------------------------------------------
    extern "C" __device__ Vec4f __direct_callable__pg_bitmap_texture_Vec4f(const Vec2f& uv, void* tex_data) {
        const BitmapTexture::Data* bitmap = reinterpret_cast<BitmapTextureData*>(tex_data);
        float4 c = tex2D<float4>(bitmap->texture, uv.x(), uv.y());
        return Vec4f(c);
    }

    extern "C" __device__ Vec3f __direct_callable__pg_bitmap_texture_Vec3f(const Vec2f & uv, void* tex_data) {
        const BitmapTexture::Data* bitmap = reinterpret_cast<BitmapTextureData*>(tex_data);
        float4 c = tex2D<float4>(bitmap->texture, uv.x(), uv.y());
        return Vec3f(c);
    }

    extern "C" __device__ Spectrum __direct_callable__pg_bitmap_texture_Spectrum(const Vec2f & uv, void* tex_data) {
        //const BitmapTexture::Data* bitmap = reinterpret_cast<BitmapTexture::Data*>(tex_data);
        //float4 c = tex2D<float4>(bitmap->texture, uv.x(), uv.y());
        //return Vec4f(c);
        return Spectrum{};
    }

    // Constant ------------------------------------------------------
    extern "C" __device__ Vec4f __direct_callable__pg_constant_texture_Vec4f(const Vec2f& uv, void* tex_data) {
        const ConstantTexture_<Vec4f>::Data* constant = reinterpret_cast<ConstantTexture_<Vec4f>::Data*>(tex_data);
        return constant->color;
    }

    extern "C" __device__ Vec3f __direct_callable__pg_constant_texture_Vec3f(const Vec2f & uv, void* tex_data) {
        const ConstantTexture_<Vec3f>::Data* constant = reinterpret_cast<ConstantTexture_<Vec4f>::Data*>(tex_data);
        return constant->color;
    }

    extern "C" __device__ Spectrum __direct_callable__pg_constant_texture_Spectrum(const Vec2f & uv, void* tex_data) {
        const ConstantTexture_<Spectrum>::Data* constant = reinterpret_cast<ConstantTexture_<Vec4f>::Data*>(tex_data);
        return constant->color;
    }

    // Checker ------------------------------------------------------
    extern "C" __device__ Vec4f __direct_callable__pg_checker_texture_Vec4f(const Vec2f& uv, void* tex_data) {
        const CheckerTexture_<Vec4f>::Data* checker = reinterpret_cast<CheckerTexture_<Vec4f>::Data*>(tex_data);
        const bool is_odd = sinf(uv.x() * math::pi * checker->scale) * sinf(uv.y() * math::pi * checker->scale) < 0;
        return is_odd ? checker->color1 : checker->color2;
    }

    extern "C" __device__ Vec4f __direct_callable__pg_checker_texture_Vec3f(const Vec2f & uv, void* tex_data) {
        const CheckerTexture_<Vec3f>::Data* checker = reinterpret_cast<CheckerTexture_<Vec4f>::Data*>(tex_data);
        const bool is_odd = sinf(uv.x() * math::pi * checker->scale) * sinf(uv.y() * math::pi * checker->scale) < 0;
        return is_odd ? checker->color1 : checker->color2;
    }

    extern "C" __device__ Vec4f __direct_callable__pg_checker_texture_Spectrum(const Vec2f & uv, void* tex_data) {
        const CheckerTexture_<Spectrum>::Data* checker = reinterpret_cast<CheckerTexture_<Spectrum>::Data*>(tex_data);
        const bool is_odd = sinf(uv.x() * math::pi * checker->scale) * sinf(uv.y() * math::pi * checker->scale) < 0;
        return is_odd ? checker->color1 : checker->color2;
    }

#endif

} // namespace prayground
