#pragma once 

#include <prayground/prayground.h>
#include "../params.h"

extern "C" {
__constant__ LaunchParams params;
}

using SurfaceInteraction = SurfaceInteraction_<Spectrum>;

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>( unpackPointer(u0, u1) ); 
}

// -------------------------------------------------------------------------------
INLINE DEVICE void trace(
    OptixTraversableHandle handle, Vec3f ro, Vec3f rd, 
    float tmin, float tmax, uint32_t ray_type, SurfaceInteraction* si) 
{
    uint32_t u0, u1;
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