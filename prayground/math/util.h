#pragma once

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
HOSTDEVICE INLINE T sqr(const T& t1)
{
    return t1 * t1;
}

template <typename T>
HOSTDEVICE INLINE T lerp(const T& a, const T& b, const float t)
{
    return a + (b - a) * t;
}

} // ::math

} // ::prayground