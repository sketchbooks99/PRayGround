#pragma once

#include <optix.h>
#include <vector_types.h>
#include <prayground/math/vec_math.h>
#include <prayground/optix/macros.h>
#include <prayground/optix/helpers.h>


/// @todo support spectrum rendering

struct RGBSpectrum
{

};

struct Spectrum
{

};

/// @todo sRGBよりlinearToSRGB の方がいい気がする
HOSTDEVICE INLINE float3 toSRGB( const float3& c )
{
    float  invGamma = 1.0f / 2.4f;
    float3 powed    = make_float3( powf( c.x, invGamma ), powf( c.y, invGamma ), powf( c.z, invGamma ) );
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f );
}

HOSTDEVICE INLINE float3 sRGBToLinear(const float3& c)
{
    const float gamma = 2.4f;
    return make_float3(
        c.x < 0.0404482f ? c.x/12.92 : powf((c.x + 0.055f)/1.055f, gamma),
        c.y < 0.0404482f ? c.y/12.92 : powf((c.y + 0.055f)/1.055f, gamma),
        c.z < 0.0404482f ? c.z/12.92 : powf((c.z + 0.055f)/1.055f, gamma)
    );
}
INLINE HOSTDEVICE float4 color2float( const uchar4& c )
{
    return make_float4(
        static_cast<float>(c.x) / 255.0f,
        static_cast<float>(c.y) / 255.0f,
        static_cast<float>(c.z) / 255.0f,
        static_cast<float>(c.w) / 255.0f       
    );
}

INLINE HOSTDEVICE float3 color2float( const uchar3& c )
{
    return make_float3(
        static_cast<float>(c.x) / 255.0f,
        static_cast<float>(c.y) / 255.0f,
        static_cast<float>(c.z) / 255.0f
    );
}

INLINE HOSTDEVICE uchar3 make_color( const float3& c, bool gamma_enalbed=true)
{
    // first apply gamma, then convert to unsigned char
    float3 rgb = c;
    if (gamma_enalbed)
        rgb = toSRGB( clamp( c, 0.0f, 1.0f ) );
    return make_uchar3( quantizeUnsigned8Bits( rgb.x ), quantizeUnsigned8Bits( rgb.y ), quantizeUnsigned8Bits( rgb.z ));
}

INLINE HOSTDEVICE uchar4 make_color( const float4& c, bool gamma_enabled=true )
{
    uchar3 rgb = make_color( make_float3( c.x, c.y, c.z ), gamma_enabled );
    return make_uchar4( rgb.x, rgb.y, rgb.z, (unsigned char)( clamp( c.w, 0.0f, 1.0f ) * 255.0f ) );
}

/**
 * @ref https://qiita.com/yoya/items/96c36b069e74398796f3
 * 
 * @note CIE XYZ
 */
HOSTDEVICE INLINE float luminance( const float3& c )
{
    return 0.299f*c.x + 0.587f*c.y + 0.114f*c.z;
}
