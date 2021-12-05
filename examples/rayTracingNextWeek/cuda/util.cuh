#pragma once 

#include <prayground/optix/cuda/device_util.cuh>
#include <prayground/math/vec_math.h>
#include <prayground/math/random.h>
#include "../params.h"

namespace prayground {

extern "C" {
__constant__ LaunchParams params;
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
    unsigned int           ray_type,
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
        ray_time,                // rayTime
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        ray_type,        
        1,           
        ray_type,        
        u0, u1 );	
}

}