#pragma once 

#include <sutil/vec_math.h>

/** MEMO: 
 * Must ray store the spectrum information?
 * */

class Ray {
public:
    SUTIL_HOSTDEVICE Ray(const float3& o, const float3&d, float t) : o(o), d(d), t(t) {}
    SUTIL_HOSTDEVICE Ray(const float3& o, const float3&d, float t, float3 c) 
    : o(o), d(d), t(t), c(c) {}

    SUTIL_HOSTDEVICE float3 origin() { return o; }
    SUTIL_HOSTDEVICE float3 direction() { return d; }
    SUTIL_HOSTDEVICE float time() { return t; }
    SUTIL_HOSTDEVICE float3 color() { return c; }
private:
    /* Position of ray origin in world coordinates. */
    float3 o;

    /* Direction of out-going ray from origin. */
    float3 d;

    /* Time of ray. It is mainly used for realizing motion blur. */
    float t;

    /* Spectrum information of ray. */
    float3 c;
}