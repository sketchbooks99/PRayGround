#pragma once 

#include <sutil/vec_math.h>

/** MEMO: 
 * Must ray store the spectrum information?
 * */

template <typename Spectrum>
class Ray {
public:
    SUTIL_HOSTDEVICE float3 origin() { return o; }
    SUTIL_HOSTDEVICE float3 direction() { return d; }
    SUTIL_HOSTDEVICE float time() { return t; }
    SUTIL_HOSTDEVICE Spectrum spectrum() { return s; }
private:
    /* Position of ray origin in world coordinates. */
    float3 o;

    /* Direction of out-going ray from origin. */
    float3 d;

    /* Time of ray. It is mainly used for realizing motion blur. */
    float t;

    /* Spectrum information of ray. */
    Spectrum s;
}