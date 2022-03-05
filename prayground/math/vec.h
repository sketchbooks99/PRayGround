#pragma once

#include <vector_types.h>
#include <prayground/optix/macros.h>

#ifndef __CUDACC__
#include <iostream>
#endif

#define VEC_DECL_T(name)              \
    using name##f  = name<float>;     \
    using name##i  = name<int32_t>;   \
    using name##c  = name<int8_t>;    \
    using name##ui = name<uint32_t>;  \
    using name##u  = name<uint8_t>;                     

namespace {
    template <typename T> struct CUVec2 {};
    template <> struct CUVec2<float> { using Type = float2; }
    template <> struct CUVec2<int32_t> { using Type = int2; }
    template <> struct CUVec2<int8_t> { using Type = char2; }
    template <> struct CUVec2<uint32_t> { using Type = uint2; }
    template <> struct CUVec2<uint8_t> { using Type = uchar2; }

    template <typename T> struct CUVec3 {};
    template <> struct CUVec3<float> { using Type = float3; }
    template <> struct CUVec3<int32_t> { using Type = int3; }
    template <> struct CUVec3<int8_t> { using Type = char3; }
    template <> struct CUVec3<uint32_t> { using Type = uint3; }
    template <> struct CUVec3<uint8_t> { using Type = uchar3; }

    template <typename T> struct CUVec4 {};
    template <> struct CUVec4<float> { using Type = float4; }
    template <> struct CUVec4<int32_t> { using Type = int4; }
    template <> struct CUVec4<int8_t> { using Type = char4; }
    template <> struct CUVec4<uint32_t> { using Type = uint4; }
    template <> struct CUVec4<uint8_t> { using Type = uchar4; }
} // nonamed namespace

namespace prayground {

    template <typename T> class Vec2;
    template <typename T> class Vec3;
    template <typename T> class Vec4;

    template <typename T>
    class Vec2 {
        using CUVec = CUVec2<T>::Type;
        using Type = T;
        using Dim = 2;

        Vec2(T x, T y) { e[0] = x; e[1] = y; }
        Vec2(T t)      { e[0] = t; e[1] = t; }
        Vec2(const CUVec& v) { e[0] = v.x; e[1] = v.y; }

              T& operator[](int32_t i)       { return e[i]; }
        const T& operator[](int32_t i) const { return e[i]; }

              T& x()       { return e[0]; }
        const T& x() const { return e[0]; }
        
              T& y()       { return e[1]; }
        const T& y() const { return e[1]; }

        const Vec2& operator-() const { return Vec2{ -e[0], -e[1] }; }

        Vec2& operator+=(const Vec2& v)
        {
            for (int i = 0; i < Dim; i++)
                e[i] += v[i];
            return *this;
        }
        Vec2& operator-=(const Vec2& v)
        {
            for (int i = 0; i < Dim; i++)
                e[i] -= v[i];
            return *this;
        }
        Vec2& operator*=(const Vec2& v)
        {
            for (int i = 0; i < Dim; i++) 
                e[i] *= v[i];
            return *this;
        }
        Vec2& operator*=(const T& t)
        {
            for (int i = 0; i < Dim; i++) 
                e[i] *= t;
            return *this;
        }
        Vec2& operator/=(const T& t)
        {
            return *this *= 1 / t;
        }
        
    private:
        T e[2];
    };

    template <typename T>
    class Vec3 {
    public:
        using CUVec = CUVec3::Type;
        using Type = T;
        using Dim = 3;

        Vec3(T x, T y, T z) { e[0] = x; e[1] = y; e[2] = z;}
        Vec3(T t) { e[0] = t; e[1] = t; e[2] = t; }

        // From other dimension vector
        Vec3(const Vec2<T>& v, const T& z) { e[0] = v[0]; e[1] = v[1]; e[2] = z; }
        Vec3(const Vec4<T>& v) { e[0] = v[0]; e[1] = v[1]; e[2] = v[2]; }

        // From CUDA vector i.e. float3
        Vec3(const CUVec2<T>::Type& v, const T& z) { e[0] = v.x; e[1] = v.y; e[2] = z; }
        Vec3(const CUVec& v) { e[0] = v.x; e[1] = v.y; e[2] = v.z; }
        Vec3(const CUVec4<T>::Type& v) { e[0] = v.x; e[1] = v.y; e[2] = v.z; }

        T& operator[](int i) { return e[i]; }
        const T& operator[](int i) const { return e[i]; }

        T& x() { return e[0]; }
        const T& x() const { return e[0]; }

        T& y() { return e[1]; }
        const T& y() const { return e[1]; }

        T& z() { return e[2]; }
        const T& z() const { return e[2]; }

        const Vec3& operator-() const { return Vec3{ -e[0], -e[1], -e[2] }; }

        Vec3& operator+=(const Vec3& v)
        {
            for (int i = 0; i < Dim; i++)
                e[i] += v[i];
            return *this;
        }
        Vec3& operator-=(const Vec3& v)
        {
            for (int i = 0; i < Dim; i++)
                e[i] -= v[i];
            return *this;
        }
        Vec3& operator*=(const Vec3& v)
        {
            for (int i = 0; i < Dim; i++)
                e[i] *= v[i];
            return *this;
        }
        Vec3& operator*=(const T& t)
        {
            for (int i = 0; i < Dim; i++)
                e[i] *= t;
            return *this;
        }
        Vec3& operator/=(const T& t)
        {
            return *this *= 1 / t;
        }

    private:
        T e[3];
    };

    template <typename T>
    class Vec4 {
    public:
        using CUVec = CUVec3::Type;
        using Type = T;
        using Dim = 3;

        Vec4(T x, T y, T z, T w) { e[0] = x; e[1] = y; e[2] = z; e[3] = w; }
        Vec4(T t) { e[0] = t; e[1] = t; e[2] = t; e[3] = t; }

        // From other dimension vector
        Vec4(const Vec2<T>& v, const T& z, const T& w) { e[0] = v[0]; e[1] = v[1]; e[2] = z; e[3] = w; }
        Vec4(const Vec2<T>& xy, const Vec2<T>& zw) { e[0] = xy[0]; e[1] = xy[1]; e[2] = zw[0]; e[3] = zw[1]; }
        Vec4(const Vec3<T>& v) { e[0] = v[0]; e[1] = v[1]; e[2] = v[2]; e[3] = T(1); }
        Vec4(const Vec3<T>& v, const T& w) { e[0] = v[0]; e[1] = v[1]; e[2] = v[2]; e[3] = w; }

        // From CUDA vector i.e. float3
        Vec4(const CUVec2<T>::Type& v, const T& z, const T& w) { e[0] = v.x; e[1] = v.y; e[2] = z; e[3] = w; }
        Vec4(const CUVec2<T>::Type& xy, const CUVec2<T>& zw) { e[0] = xy.x; e[1] = xy.y; e[2] = zw.x; e[3] = zw.y; }
        Vec4(const CUVec3<T>::Type& v) { e[0] = v.x; e[1] = v.y; e[2] = v.z; e[3] = T(1); }
        Vec4(const CUVec3<T>::Type& v, const T& w) { e[0] = v.x; e[1] = v.y; e[2] = v.z; e[3] = w; }
        Vec4(const CUVec& v) { e[0] = v.x; e[1] = v.y; e[2] = v.z; e[3] = v.w; }

        T& operator[](int i) { return e[i]; }
        const T& operator[](int i) const { return e[i]; }

        T& x() { return e[0]; }
        const T& x() const { return e[0]; }

        T& y() { return e[1]; }
        const T& y() const { return e[1]; }

        T& z() { return e[2]; }
        const T& z() const { return e[2]; }

        T& w() { return e[3]; }
        const T& w() const { return e[3]; }

        const Vec4& operator-() const { return Vec4{ -e[0], -e[1], -e[2] }; }

        Vec4& operator+=(const Vec4& v)
        {
            for (int i = 0; i < Dim; i++)
                e[i] += v[i];
            return *this;
        }
        Vec4& operator-=(const Vec4& v)
        {
            for (int i = 0; i < Dim; i++)
                e[i] -= v[i];
            return *this;
        }
        Vec4& operator*=(const Vec4& v)
        {
            for (int i = 0; i < Dim; i++)
                e[i] *= v[i];
            return *this;
        }
        Vec4& operator*=(const T& t)
        {
            for (int i = 0; i < Dim; i++)
                e[i] *= t;
            return *this;
        }
        Vec4& operator/=(const T& t)
        {
            return *this *= 1 / t;
        }
    private:
        T e[4];
    };

    // ----------------------------------------------------------------------
    // Utilities for Vec2
    // ----------------------------------------------------------------------
    template <typename T> 
    inline HOSTDEVICE Vec2<T> operator+(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return Vec2<T>{v1[0] + v2[0], v1[1] + v2[1]};
    }

    template <typename T>
    inline HOSTDEVICE Vec2<T> operator-(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return Vec2<T>{v1[0] - v2[0], v1[1] - v2[1]};
    }

    template <typename T> 
    inline HOSTDEVICE Vec2<T> operator*(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return Vec2<T>{v1[0] * v2[0], v1[1] * v2[1]};
    }

    template <typename T> 
    inline HOSTDEVICE Vec2<T> operator*(const T& t, const Vec2<T>& v)
    {
        return Vec2<T>{v[0] * t, v[1] * t};
    }

    template <typename T>
    inline HOSTDEVICE Vec2<T> operator*(const Vec2<T>& v, const T& t)
    {
        return t * v;
    }

    template <typename T> 
    inline HOSTDEVICE Vec2<T> operator/(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return Vec2<T>{v1[0] / v2[0], v1[1] / v2[1]};
    }

    template <typename T>
    inline HOSTDEVICE Vec2<T> operator/(const Vec2<T>& v, const T& t)
    {
        return v * (1 / t);
    }

    template <typename T> 
    inline HOSTDEVICE T dot(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return v1[0] * v2[0] + v1[1] * v2[1];
    }

    template <typename T> 
    inline HOSTDEVICE T cross(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return v1[0] * v2[1] - v1[1] * v2[0];
    }

    template <typename T>
    inline HOSTDEVICE Vec2<T> length(const Vec2<T>& v)
    {
        return sqrtf(lengthSquared(v));
    }

    template <typename T>
    inline HOSTDEVICE Vec2<T> lengthSquared(const Vec2<T>& v)
    {
        return dot(v, v);
    }

    template <typename T> 
    inline HOSTDEVICE Vec2<T> normalize(const Vec2<T>& v)
    {
        return v / length(v);
    }

    // ----------------------------------------------------------------------
    // Utilities for Vec3
    // ----------------------------------------------------------------------
    template <typename T>
    inline HOSTDEVICE Vec3<T> operator+(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>{v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]};
    }

    template <typename T>
    inline HOSTDEVICE Vec3<T> operator-(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>{v1[0] - v2[0], v1[1] - v2[1], v1[1] - v2[2]};
    }

    template <typename T>
    inline HOSTDEVICE Vec3<T> operator*(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>{v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]};
    }

    template <typename T>
    inline HOSTDEVICE Vec3<T> operator*(const T& t, const Vec3<T>& v)
    {
        return Vec3<T>{v[0] * t, v[1] * t, v[2] * t};
    }

    template <typename T>
    inline HOSTDEVICE Vec3<T> operator*(const Vec3<T>& v, const T& t)
    {
        return t * v;
    }

    template <typename T>
    inline HOSTDEVICE Vec3<T> operator/(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>{v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2]};
    }

    template <typename T>
    inline HOSTDEVICE Vec3<T> operator/(const Vec3<T>& v, const T& t)
    {
        return v * (1 / t);
    }

    template <typename T>
    inline HOSTDEVICE T dot(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    }

    template <typename T>
    inline HOSTDEVICE T cross(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>{
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        };
    }

    template <typename T>
    inline HOSTDEVICE Vec3<T> length(const Vec3<T>& v)
    {
        return sqrtf(lengthSquared(v));
    }

    template <typename T>
    inline HOSTDEVICE Vec3<T> lengthSquared(const Vec3<T>& v)
    {
        return dot(v, v);
    }

    template <typename T>
    inline HOSTDEVICE Vec3<T> normalize(const Vec3<T>& v)
    {
        return v / length(v);
    }

    // ----------------------------------------------------------------------
    // Utilities for Vec4
    // ----------------------------------------------------------------------
    template <typename T>
    inline HOSTDEVICE Vec4<T> operator+(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>{v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2], v1[3] + v2[3]};
    }

    template <typename T>
    inline HOSTDEVICE Vec4<T> operator-(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>{v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2], v1[3] - v2[3]};
    }

    template <typename T>
    inline HOSTDEVICE Vec4<T> operator*(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>{v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2], v1[3] * v2[3]};
    }

    template <typename T>
    inline HOSTDEVICE Vec4<T> operator*(const T& t, const Vec4<T>& v)
    {
        return Vec4<T>{v[0] * t, v[1] * t, v[2] * t, v[3] * t};
    }

    template <typename T>
    inline HOSTDEVICE Vec4<T> operator*(const Vec4<T>& v, const T& t)
    {
        return t * v;
    }

    template <typename T>
    inline HOSTDEVICE Vec4<T> operator/(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>{v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2], v1[3] / v2[3]};
    }

    template <typename T>
    inline HOSTDEVICE Vec4<T> operator/(const Vec4<T>& v, const T& t)
    {
        return v * (1 / t);
    }

    template <typename T>
    inline HOSTDEVICE T dot(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    }

    template <typename T>
    inline HOSTDEVICE T cross(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>{
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0], 
            1.0f
        };
    }

    template <typename T>
    inline HOSTDEVICE Vec4<T> length(const Vec4<T>& v)
    {
        return sqrtf(lengthSquared(v));
    }

    template <typename T>
    inline HOSTDEVICE Vec4<T> lengthSquared(const Vec4<T>& v)
    {
        return dot(v, v);
    }

    template <typename T>
    inline HOSTDEVICE Vec4<T> normalize(const Vec4<T>& v)
    {
        return v / length(v);
    }


    VEC_DECL_T(Vec2)
    VEC_DECL_T(Vec3)
    VEC_DECL_T(Vec4)
}