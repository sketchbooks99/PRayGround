#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <prayground/core/interaction.h>
#include <prayground/optix/helpers.h>
#include <prayground/optix/macros.h>

namespace prayground {

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

template <typename T>
INLINE DEVICE T* getPayloadPtr()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
}

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<SurfaceInteraction*>( unpackPointer(u0, u1) ); 
}

// -------------------------------------------------------------------------------
static INLINE DEVICE void setPayloadOcclusion(bool occluded)
{
	optixSetPayload_0(static_cast<unsigned int>(occluded));
}

INLINE DEVICE void trace(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    unsigned int           ray_type,
    unsigned int           ray_count,
    SurfaceInteraction*    si
) 
{
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
        ray_type,        
        ray_count,           
        ray_type,        
        u0, u1 );	
}

} // ::prayground