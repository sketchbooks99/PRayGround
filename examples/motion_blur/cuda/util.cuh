#pragma once 

#include <optix.h>
#include <cuda_runtime.h>
#include <prayground/math/vec_math.h>
#include <prayground/math/util.h>
#include <prayground/math/random.h>
#include <prayground/optix/util.h>
#include <prayground/optix/macros.h>
#include "../params.h"

namespace prayground {

extern "C" {
__constant__ LaunchParams params;
}

struct SurfaceInteraction
{
    float3 p;
    float3 n;
    float3 albedo;
    float3 shading_val;
    float2 uv;
};

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

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<SurfaceInteraction*>( unpackPointer(u0, u1) ); 
}

// -------------------------------------------------------------------------------
INLINE DEVICE void trace(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    float                  ray_time, 
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
        ray_time,          // ray time
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        0,        
        1,           
        0,        
        u0, u1 );	
}

} // ::prayground