#pragma once 

#include <sutil/vec_math.h>
#include <include/optix/macros.h>

/** MEMO: 
 * Must ray store the spectrum information?
 * */

#ifdef __CUDACC__

namespace pt {

class Ray {
public:
    DEVICE Ray(const float3& o, const float3&d, float t) : o(o), d(d), t(t) {}
    DEVICE Ray(const float3& o, const float3&d, float t, float3 c) 
    : o(o), d(d), t(t), c(c) {}

    DEVICE float3 origin() const { return o; }
    DEVICE float3 direction() const { return d; }
    DEVICE float time() const { return t; }
    DEVICE float color() const { return c; }

private:
    /* Position of ray origin in world coordinates. */
    float3 o;

    /* Direction of out-going ray from origin. */
    float3 d;

    /* Time of ray. It is mainly used for realizing motion blur. */
    float t;

    /* Spectrum information of ray. */
    float3 c;
};

}

#endif