#pragma once 

#include <optix.h>
#include <cuda_runtime.h>
#include <prayground/math/vec_math.h>
#include <prayground/math/util.h>
#include <prayground/optix/helpers.h>
#include <prayground/optix/macros.h>
#include "../params.h"

namespace prayground {

extern "C" {
__constant__ LaunchParams params;
}

static DEVICE void init_rand_state(SurfaceInteraction* si, uint2 launch_dim, uint3 launch_idx, unsigned int frame)
{
    curandState_t state;
    si->curand_state = &state;
    curand_init(launch_idx.y * launch_dim.x + launch_idx.x, frame, 0, si->curand_state);
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

INLINE DEVICE prayground::SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<prayground::SurfaceInteraction*>( unpackPointer(u0, u1) ); 
}

// -------------------------------------------------------------------------------
INLINE DEVICE void trace(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    unsigned int           ray_type,
    prayground::SurfaceInteraction*    si
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
        1,           
        ray_type,        
        u0, u1 );	
}

}