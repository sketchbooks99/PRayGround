#pragma once

#include <vector_functions.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

#ifndef __CUDACC_RTC__
#include <cmath>
#include <cstdlib>
#endif

namespace prayground {

enum class Axis {
    X = 0, 
    Y = 1, 
    Z = 2
};

namespace math {

constexpr float pi = 3.14159265358979323846f;
constexpr float two_pi = 6.283185307179586232f;
constexpr float inv_pi = 0.3183098861837906912164f;
constexpr float eps = 1e-10f;

HOSTDEVICE INLINE float radians(const float degrees)
{
    return degrees * pi / 180.0f;
}

HOSTDEVICE INLINE float degrees(const float radians)
{
    return radians * 180.0f / pi;
}

template <typename T>
HOSTDEVICE INLINE T random(T _min, T _max, curandState_t* state)
{
    return static_cast<T>(_min + curand_uniform(state) * (float)(_max+1 - _min));
}

template <typename T>
HOSTDEVICE INLINE T sqr(T t1)
{
    return t1 * t1;
}

} // ::math

} // ::prayground