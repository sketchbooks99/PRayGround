#pragma once 

#include <prayground/math/vec.h>
#include <prayground/optix/macros.h>

namespace prayground {

    struct Ray {
        HOSTDEVICE INLINE Vec3f at(const float time) { return o + d*time; }

        /* Position of ray origin in world coordinates. */
        Vec3f o;

        /* Direction of out-going ray from origin. */
        Vec3f d;

        /* Time of ray. It is mainly used for realizing motion blur. */
        float tmin;
        float tmax;
        float t;
    };

    struct pRay {
        /** @todo Polarized ray */
        HOSTDEVICE INLINE Vec3f at(const float time) { return o + d*time; }

        /* Position of ray origin in world coordinates. */
        Vec3f o;

        /* Direction of out-going ray from origin. */
        Vec3f d; 

        /* Tangent vector along with ray direction */
        Vec3f tangent;

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

} // namespace prayground