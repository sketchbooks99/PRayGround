#pragma once 

#include <prayground/prayground.h>
#include "../params.h"

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

struct LightInteraction
{
    Vec3f p;
    Vec3f n;
    Vec2f uv;
    Vec3f wi;
    float pdf;
};

extern "C" { __constant__ LaunchParams params; }

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<SurfaceInteraction*>( unpackPointer(u0, u1) ); 
}

// -------------------------------------------------------------------------------
INLINE DEVICE void trace(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd, 
    float tmin, float tmax, uint32_t ray_type, SurfaceInteraction* si) 
{
    uint32_t u0, u1;
    packPointer( si, u0, u1 );
    optixTrace(
        handle, ro, rd,
        tmin, tmax, 0.0f, 
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        ray_type, 2, ray_type,        
        u0, u1 );	
}

INLINE DEVICE bool traceShadow(OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd, float tmin, float tmax)
{
    uint32_t is_hit;
    optixTrace(
        handle,
        ro,
        rd,
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        1, 2, 1,
        is_hit
    );
    return static_cast<bool>(is_hit);
}