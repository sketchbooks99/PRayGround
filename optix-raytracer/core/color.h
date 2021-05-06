#pragma once

#include <optix.h>
#include <vector_types.h>
#include <sutil/vec_math.h>
#include "../optix/macros.h"
#include "../optix/helpers.h"

HOSTDEVICE INLINE float3 toSRGB( const float3& c )
{
    float  invGamma = 1.0f / 2.4f;
    float3 powed    = make_float3( powf( c.x, invGamma ), powf( c.y, invGamma ), powf( c.z, invGamma ) );
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f );
}

HOSTDEVICE INLINE uchar4 make_color( const float3& c )
{
    // first apply gamma, then convert to unsigned char
    float3 srgb = toSRGB( clamp( c, 0.0f, 1.0f ) );
    return make_uchar4( quantizeUnsigned8Bits( srgb.x ), quantizeUnsigned8Bits( srgb.y ), quantizeUnsigned8Bits( srgb.z ), 255u );
}

HOSTDEVICE INLINE uchar4 make_color( const float4& c )
{
    return make_color( make_float3( c.x, c.y, c.z ) );
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
