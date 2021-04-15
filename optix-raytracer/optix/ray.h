#pragma once 

#include <sutil/vec_math.h>
#include "../optix/macros.h"

/** MEMO: 
 * Must ray store the spectrum information?
 * */

#ifdef __CUDACC__

namespace oprt {

struct Ray {

    float3 at(float time) { return o + d*time; }

    /* Position of ray origin in world coordinates. */
    float3 o;

    /* Direction of out-going ray from origin. */
    float3 d;

    /* Time of ray. It is mainly used for realizing motion blur. */
    float tmin;
    float tmax;
    float t;

    /* Spectrum information of ray. */
    float3 spectrum;
};

}

INLINE DEVICE oprt::Ray get_ray() {
    oprt::Ray ray;
    ray.o = optixTransformPointFromWorldToObjectSpace(optixGetWorldRayOrigin());
    ray.d = optixTransformVectorFromWorldToObjectSpace(optixGetWorldRayDirection());
    ray.tmin = optixGetRayTmin();
    ray.tmax = optixGetRayTmax();
    ray.t = optixGetRayTime();
    return ray;
}

#endif