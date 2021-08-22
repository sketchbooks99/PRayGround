#pragma once

#include <vector_functions.h>
#include <vector_types.h>

#ifndef __CUDACC_RTC__
#include <cmath>
#include <cstdlib>
#endif

namespace oprt {

enum class Axis {
    X = 0, 
    Y = 1, 
    Z = 2
};

namespace constants {

constexpr float pi = 3.14159265358979323846f;
constexpr float two_pi = 6.283185307179586232f;
constexpr float inv_pi = 0.3183098861837906912164f;
constexpr float eps = 1e-10f;

} // ::constant
} // ::oprt