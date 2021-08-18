#pragma once 

#include <optix.h>
#include <cuda_runtime.h>
#include "../../oprt/optix/helpers.h"
#include "../../oprt/core/util.h"

namespace oprt {

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
    float3 attenuation;
    float3 emission;

    /** UV coordinate at an intersection point. */
    float2 uv;

    /** Derivatives on texture coordinates. */
    float3 dpdu;    // Tangent vector at a surface.
    float3 dpdv;    // Binormal vector at a surface.

    /** Seed for random */
    unsigned int seed;

    SurfaceProperty surface_property;

    SurfaceType surface_type;

    bool trace_terminate;
    bool radiance_evaled;
};

extern "C"
{
__constant__ LaunchParams params;
}

template <typename T>
__forceinline__ __device__ swap(T& a, T& b)
{
    T c(a); a = b; b = c;
}

__forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

__forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

__forceinline__ __device__ oprt::SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<oprt::SurfaceInteraction*>( unpackPointer(u0, u1) ); 
}


INLINE DEVICE void trace(
    OptixTraversableHandle handle, 
    float3 ro, float3 rd, float tmin, float tmax,
    SurfaceInteraction* si
) 
{
    unsigned int u0, u1;
    packPointer( si, u0, u1 );
    optixTrace(
        handle,
        ro,
        rd,
        tmin,
        tmax,
        0.0f,                // rayTime
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        0,  // SBToffset 
        1,  // SBTstride
        0,  // missSBTIndex    
        u0, u1 );	
}

} // ::oprt