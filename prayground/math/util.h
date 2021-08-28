#pragma once

#include <vector_functions.h>
#include <vector_types.h>

#ifndef __CUDACC_RTC__
#include <cmath>
#include <cstdlib>
#endif

namespace prayground {

namespace constants {

constexpr float pi = 3.14159265358979323846f;
constexpr float two_pi = 6.283185307179586232f;
constexpr float inv_pi = 0.3183098861837906912164f;
constexpr float eps = 1e-10f;

} // ::constant

enum class Axis {
    X = 0, 
    Y = 1, 
    Z = 2
};

HOSTDEVICE INLINE float radians(const float degrees)
{
    return degrees * constants::pi / 180.0f;
}

HOSTDEVICE INLINE float degrees(const float radians)
{
    return radians * 180.0f / constants::pi;
}

} // ::prayground