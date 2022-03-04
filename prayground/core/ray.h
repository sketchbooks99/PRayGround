#pragma once 

#include <prayground/math/vec_math.h>
#include <prayground/optix/macros.h>

namespace prayground {

struct Ray {
    HOSTDEVICE INLINE float3 at(const float time) { return o + d*time; }

    /* Position of ray origin in world coordinates. */
    float3 o;

    /* Direction of out-going ray from origin. */
    float3 d;

    /* Time of ray. It is mainly used for realizing motion blur. */
    float tmin;
    float tmax;
    float t;
};

struct pRay {
    /** @todo Polarized ray */
    HOSTDEVICE INLINE float3 at(const float time) { return o + d*time; }

    /* Position of ray origin in world coordinates. */
    float3 o;

    /* Direction of out-going ray from origin. */
    float3 d; 

    /* Tangent vector along with ray direction */
    float3 tangent;

    /* Time of ray. It is mainly used for realizing motion blur. */
    float tmin; 
    float tmax; 
    float t;
};

/** Useful function to get ray info on OptiX */
#ifdef __CUDACC__

INLINE DEVICE Ray getLocalRay() {
    Ray ray;
    ray.o = optixTransformPointFromWorldToObjectSpace( optixGetWorldRayOrigin() );
    ray.d = optixTransformVectorFromWorldToObjectSpace( optixGetWorldRayDirection() );
    ray.tmin = optixGetRayTmin();
    ray.tmax = optixGetRayTmax();
    ray.t = optixGetRayTime();
    return ray;
}

INLINE DEVICE Ray getWorldRay() {
    Ray ray;
    ray.o = optixGetWorldRayOrigin();
    ray.d = optixGetWorldRayDirection();
    ray.tmin = optixGetRayTmin();
    ray.tmax = optixGetRayTmax();
    ray.t = optixGetRayTime();
    return ray;
}

INLINE DEVICE pRay getLocalpRay() 
{
    
}

INLINE DEVICE pRay getWorldpRay() 
{

}

#endif

} // ::prayground