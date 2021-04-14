#pragma once

#include <optix.h>
#include "../optix/helpers.h"
#include "../optix/macros.h"

enum RayType {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT = 2
};

namespace pt {

/// @note Currently \c spectrum is RGB representation, not spectrum. 
struct SurfaceInteraction {
    /** Position of intersection point in world coordinates. */
    float3 p;

    /** Surface normal of primitive at an intersection point. */
    float3 n;

    /** Incident and outgoing directions at a surface. */
    float3 wi;
    float3 wo;

    /** Spectrum information of ray. */
    float3 spectrum;

    /** Radiance and attenuation computed by a material attached with a surface. */
    float3 radiance;
    float3 attenuation;
    float3 emission;

    /** UV coordinate at an intersection point. */
    float2 uv;

    /** Derivatives on texture coordinates. */
    float3 dpdu;    // Tangent vector at a surface.
    float3 dpdv;    // Binormal vector at a surface.

    /** Seed for random */
    unsigned int seed;

    int trace_terminate;
};

}

#ifdef __CUDACC__
template <typename T>
INLINE DEVICE void swap(T& a, T& b)
{
    T c(a); a = b; b = c;
}

INLINE DEVICE void* unpack_pointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void* ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}

INLINE DEVICE void pack_pointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

INLINE DEVICE pt::SurfaceInteraction* get_surfaceinteraction()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<pt::SurfaceInteraction*>( unpack_pointer(u0, u1) ); 
}
#endif
