#pragma once

#include <vector_functions.h>
#include <prayground/optix/macros.h>

#ifndef __CUDACC__
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

    } // namespace math

    template <typename T>
    HOSTDEVICE INLINE T pow2(const T& t)
    {
        return t * t;
    }

    template <typename T>
    HOSTDEVICE INLINE T pow3(const T& t)
    {
        return t * t * t;
    }

    template <typename T>
    HOSTDEVICE INLINE T pow4(const T& t)
    {
        return t * t * t * t;
    }

    template <typename T>
    HOSTDEVICE INLINE T pow5(const T& t)
    {
        return t * t * t * t * t;
    }

    template <typename T>
    HOSTDEVICE INLINE T lerp(const T& a, const T& b, const float t)
    {
        return a + (b - a) * t;
    }

    HOSTDEVICE INLINE float clamp(const float f, const float a, const float b)
    {
        return fmaxf(a, fminf(f, b));
    }

    template <typename IntegerType>
    INLINE HOSTDEVICE IntegerType roundUp(IntegerType x, IntegerType y)
    {
        return ( ( x + y - 1 ) / y ) * y;
    }

    template <typename T>
    HOSTDEVICE INLINE T max(const T& a, const T& b)
    {
        return a > b ? a : b;
    }

    template <typename T>
    HOSTDEVICE INLINE T min(const T& a, const T& b)
    {
        return a < b ? a : b;
    }

} // namespace prayground