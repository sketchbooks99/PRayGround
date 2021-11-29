#pragma once 

#include <optix.h>
#include <cuda_runtime.h>
#include <prayground/math/vec_math.h>
#include <prayground/math/util.h>
#include <prayground/math/random.h>
#include <prayground/optix/helpers.h>
#include <prayground/optix/macros.h>
#include <prayground/optix/cuda/device_util.cuh>
#include "../params.h"

namespace prayground {

extern "C" {
__constant__ LaunchParams params;
}

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = getPayload<0>();
    const unsigned int u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>( unpackPointer(u0, u1) ); 
}

// -------------------------------------------------------------------------------
INLINE DEVICE void trace(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
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
        0.0f,                // rayTime
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        ray_type,        
        1,           
        ray_type,        
        u0, u1 );	
}

}