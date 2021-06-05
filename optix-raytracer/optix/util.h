#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include "helpers.h"
#include "macros.h"
#include "../optix-raytracer.h"

enum RayType {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT = 2
};

namespace oprt {

struct MaterialProperty
{
    void* matdata;
    unsigned int bsdf_sample_id;
    unsigned int pdf_id;
};

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

    MaterialProperty mat_property;

    int trace_terminate;
    int radiance_evaled;
};

}

#ifdef __CUDACC__

extern "C" {
__constant__ oprt::Params params;
}

template <typename T>
INLINE DEVICE void swap(T& a, T& b)
{
    T c(a); a = b; b = c;
}

INLINE DEVICE void* unpackPointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void* ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}

INLINE DEVICE void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

INLINE DEVICE oprt::SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<oprt::SurfaceInteraction*>( unpackPointer(u0, u1) ); 
}

// -------------------------------------------------------------------------------
static INLINE DEVICE void setPayloadOcclusion(bool occluded)
{
	optixSetPayload_0(static_cast<unsigned int>(occluded));
}

INLINE DEVICE bool traceOcclusion(
    OptixTraversableHandle handle, float3 ro, float3 rd, float tmin, float tmax
) 
{
    unsigned int occluded = 0u;
    optixTrace(
        handle, 
        ro, 
        rd, 
        tmin, 
        tmax,
        0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        RAY_TYPE_OCCLUSION,
        RAY_TYPE_COUNT,
        RAY_TYPE_OCCLUSION,
        occluded
    );
    return occluded;
}

INLINE DEVICE void traceRadiance(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    oprt::SurfaceInteraction*    si
) 
{
    // TODO: deduce stride from num ray-types passed in params

    unsigned int u0, u1;
    packPointer( si, u0, u1 );
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,                // rayTime
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_RADIANCE,        
        RAY_TYPE_COUNT,           
        RAY_TYPE_RADIANCE,        
        u0, u1 );	
}

#endif
