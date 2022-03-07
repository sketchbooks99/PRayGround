#pragma once

#include <vector_types.h>
#include <vector_functions.h>
#include <prayground/optix/macros.h>

#ifndef __CUDACC__
#include <iostream>
#include <cmath>
#include <cstdlib>
#endif // __CUDACC__

#ifdef __CUDACC__
#include <prayground/optix/cuda/device_util.cuh>
#endif

#define VEC_DECL_T(name)               \
    using name##f   = name<float>;     \
    using name##d   = name<double>;    \
    using name##c   = name<int8_t>;    \
    using name##s   = name<int16_t>;   \
    using name##i   = name<int32_t>;   \
    using name##ll  = name<int64_t>;   \
    using name##u   = name<uint8_t>;   \
    using name##us  = name<uint16_t>;  \
    using name##ui  = name<uint32_t>;  \
    using name##ull = name<uint64_t>;  

#define CUVEC2_DECL_ALIAS(vname, T) \
    using Type = vname;             \
    static constexpr vname (*makeV)(T, T) = &make_##vname;

#define CUVEC3_DECL_ALIAS(vname, T) \
    using Type = vname;             \
    static constexpr vname (*makeV)(T, T, T) = &make_##vname;

#define CUVEC4_DECL_ALIAS(vname, T) \
    using Type = vname;             \
    static constexpr vname (*makeV)(T, T, T, T) = &make_##vname;

namespace prayground {

    template <typename T> struct CUVec2 {};
    template <> struct CUVec2<float>    { CUVEC2_DECL_ALIAS(float2, float) };
    template <> struct CUVec2<double>   { CUVEC2_DECL_ALIAS(double2, double) };
    template <> struct CUVec2<int8_t>   { CUVEC2_DECL_ALIAS(char2, signed char) };
    template <> struct CUVec2<int16_t>  { CUVEC2_DECL_ALIAS(short2, short) };
    template <> struct CUVec2<int32_t>  { CUVEC2_DECL_ALIAS(int2, int) };
    template <> struct CUVec2<int64_t>  { CUVEC2_DECL_ALIAS(longlong2, long long int) };
    template <> struct CUVec2<uint8_t>  { CUVEC2_DECL_ALIAS(uchar2, unsigned char) };
    template <> struct CUVec2<uint16_t> { CUVEC2_DECL_ALIAS(ushort2, unsigned short) };
    template <> struct CUVec2<uint32_t> { CUVEC2_DECL_ALIAS(uint2, unsigned int) };
    template <> struct CUVec2<uint64_t> { CUVEC2_DECL_ALIAS(ulonglong2, unsigned long long int) };

    template <typename T> struct CUVec3 {};
    template <> struct CUVec3<float>    { CUVEC3_DECL_ALIAS(float3, float) };
    template <> struct CUVec3<double>   { CUVEC3_DECL_ALIAS(double3, double) };
    template <> struct CUVec3<int8_t>   { CUVEC3_DECL_ALIAS(char3, signed char) };
    template <> struct CUVec3<int16_t>  { CUVEC3_DECL_ALIAS(short3, short) };
    template <> struct CUVec3<int32_t>  { CUVEC3_DECL_ALIAS(int3, int) };
    template <> struct CUVec3<int64_t>  { CUVEC3_DECL_ALIAS(longlong3, long long int) };
    template <> struct CUVec3<uint8_t>  { CUVEC3_DECL_ALIAS(uchar3, unsigned char) };
    template <> struct CUVec3<uint16_t> { CUVEC3_DECL_ALIAS(ushort3, unsigned short) };
    template <> struct CUVec3<uint32_t> { CUVEC3_DECL_ALIAS(uint3, unsigned int) };
    template <> struct CUVec3<uint64_t> { CUVEC3_DECL_ALIAS(ulonglong3, unsigned long long int) };

    template <typename T> struct CUVec4 {};
    template <> struct CUVec4<float>    { CUVEC4_DECL_ALIAS(float4, float) };
    template <> struct CUVec4<double>   { CUVEC4_DECL_ALIAS(double4, double) };
    template <> struct CUVec4<int8_t>   { CUVEC4_DECL_ALIAS(char4, signed char) };
    template <> struct CUVec4<int16_t>  { CUVEC4_DECL_ALIAS(short4, short) };
    template <> struct CUVec4<int32_t>  { CUVEC4_DECL_ALIAS(int4, int) };
    template <> struct CUVec4<int64_t>  { CUVEC4_DECL_ALIAS(longlong4, long long int) };
    template <> struct CUVec4<uint8_t>  { CUVEC4_DECL_ALIAS(uchar4, unsigned char) };
    template <> struct CUVec4<uint16_t> { CUVEC4_DECL_ALIAS(ushort4, unsigned short) };
    template <> struct CUVec4<uint32_t> { CUVEC4_DECL_ALIAS(uint4, unsigned int) };
    template <> struct CUVec4<uint64_t> { CUVEC4_DECL_ALIAS(ulonglong4, unsigned long long int) };

    template <typename T> class Vec2;
    template <typename T> class Vec3;
    template <typename T> class Vec4;

    template <typename T>
    class __align__(sizeof(T) * 2) Vec2 {
    public:
        using CUVec = typename CUVec2<T>::Type;
        using Type = T;
        static constexpr uint32_t Dim = 2;

        Vec2(T x, T y) { e[0] = x; e[1] = y; }
        Vec2(T t = T(0)) { e[0] = t; e[1] = t; }
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
            for (uint32_t i = 0; i < Dim; i++)
                e[i] += v[i];
            return *this;
        }
        Vec2& operator-=(const Vec2& v)
        {
            for (uint32_t i = 0; i < Dim; i++)
                e[i] -= v[i];
            return *this;
        }
        Vec2& operator*=(const Vec2& v)
        {
            for (uint32_t i = 0; i < Dim; i++) 
                e[i] *= v[i];
            return *this;
        }
        Vec2& operator*=(const T& t)
        {
            for (uint32_t i = 0; i < Dim; i++) 
                e[i] *= t;
            return *this;
        }
        Vec2& operator/=(const T& t)
        {
            return *this *= 1 / t;
        }

        CUVec toCUVec() const { return CUVec2<T>::makeV(e[0], e[1]); }
        
    private:
        T e[2];
    };

    template <typename T>
    class Vec3 {
    public:
        using CUVec = typename CUVec3<T>::Type;
        using Type = T;
        static constexpr uint32_t Dim = 3;

        Vec3(T x, T y, T z) { e[0] = x; e[1] = y; e[2] = z;}
        Vec3(T t = T(0)) { e[0] = t; e[1] = t; e[2] = t; }

        // From other dimension vector
        Vec3(const Vec2<T>& v, const T& z) { e[0] = v[0]; e[1] = v[1]; e[2] = z; }
        Vec3(const Vec4<T>& v) { e[0] = v[0]; e[1] = v[1]; e[2] = v[2]; }

        // From CUDA vector i.e. float3
        Vec3(const typename CUVec2<T>::Type& v, const T& z) { e[0] = v.x; e[1] = v.y; e[2] = z; }
        Vec3(const CUVec& v) { e[0] = v.x; e[1] = v.y; e[2] = v.z; }
        Vec3(const typename CUVec4<T>::Type& v) { e[0] = v.x; e[1] = v.y; e[2] = v.z; }

        T& operator[](int i) { return e[i]; }
        const T& operator[](int i) const { return e[i]; }

        T& x() { return e[0]; }
        const T& x() const { return e[0]; }

        T& y() { return e[1]; }
        const T& y() const { return e[1]; }

        T& z() { return e[2]; }
        const T& z() const { return e[2]; }

        Vec3 operator-() const { return Vec3{ -e[0], -e[1], -e[2] }; }

        Vec3& operator+=(const Vec3& v)
        {
            for (uint32_t i = 0; i < Dim; i++)
                e[i] += v[i];
            return *this;
        }
        Vec3& operator-=(const Vec3& v)
        {
            for (uint32_t i = 0; i < Dim; i++)
                e[i] -= v[i];
            return *this;
        }
        Vec3& operator*=(const Vec3& v)
        {
            for (uint32_t i = 0; i < Dim; i++)
                e[i] *= v[i];
            return *this;
        }
        Vec3& operator*=(const float t)
        {
            for (uint32_t i = 0; i < Dim; i++)
                e[i] *= t;
            return *this;
        }
        Vec3& operator/=(const float t)
        {
            return *this *= 1 / t;
        }

        CUVec toCUVec() const { return CUVec3<T>::makeV(e[0], e[1], e[2]); }

    private:
        T e[3];
    };

    template <typename T>
    class __align__(sizeof(T) * 4) Vec4 {
    public:
        using CUVec = typename CUVec4<T>::Type;
        using Type = T;
        static constexpr uint32_t Dim = 4;

        Vec4(T x, T y, T z, T w) { e[0] = x; e[1] = y; e[2] = z; e[3] = w; }
        Vec4(T t = T(0)) { e[0] = t; e[1] = t; e[2] = t; e[3] = t; }

        // From other dimension vector
        Vec4(const Vec2<T>& v, const T& z, const T& w) { e[0] = v[0]; e[1] = v[1]; e[2] = z; e[3] = w; }
        Vec4(const Vec2<T>& xy, const Vec2<T>& zw) { e[0] = xy[0]; e[1] = xy[1]; e[2] = zw[0]; e[3] = zw[1]; }
        Vec4(const Vec3<T>& v) { e[0] = v[0]; e[1] = v[1]; e[2] = v[2]; e[3] = T(1); }
        Vec4(const Vec3<T>& v, const T& w) { e[0] = v[0]; e[1] = v[1]; e[2] = v[2]; e[3] = w; }

        // From CUDA vector i.e. float3
        Vec4(const typename CUVec2<T>::Type& v, const T& z, const T& w) { e[0] = v.x; e[1] = v.y; e[2] = z; e[3] = w; }
        Vec4(const typename CUVec2<T>::Type& xy, const typename CUVec2<T>::Type& zw) { e[0] = xy.x; e[1] = xy.y; e[2] = zw.x; e[3] = zw.y; }
        Vec4(const typename CUVec3<T>::Type& v) { e[0] = v.x; e[1] = v.y; e[2] = v.z; e[3] = T(1); }
        Vec4(const typename CUVec3<T>::Type& v, const T& w) { e[0] = v.x; e[1] = v.y; e[2] = v.z; e[3] = w; }
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
            for (uint32_t i = 0; i < Dim; i++)
                e[i] += v[i];
            return *this;
        }
        Vec4& operator-=(const Vec4& v)
        {
            for (uint32_t i = 0; i < Dim; i++)
                e[i] -= v[i];
            return *this;
        }
        Vec4& operator*=(const Vec4& v)
        {
            for (uint32_t i = 0; i < Dim; i++)
                e[i] *= v[i];
            return *this;
        }
        Vec4& operator*=(const float t)
        {
            for (uint32_t i = 0; i < Dim; i++)
                e[i] *= t;
            return *this;
        }
        Vec4& operator/=(const float t)
        {
            return *this *= 1 / t;
        }

        CUVec toCUVec() const { return CUVec4<T>::makeV(e[0], e[1], e[2], e[3]); }
    private:
        T e[4];
    };

    // ----------------------------------------------------------------------
    // Utilities for Vec2
    // ----------------------------------------------------------------------
#ifndef __CUDACC__
    template <typename T>
    INLINE std::ostream& operator<<(std::ostream& out, const Vec2<T>& v)
    {
        return out << v.x() << ' ' << v.y();
    }
#endif

    template <typename T> 
    INLINE HOSTDEVICE Vec2<T> operator+(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return Vec2<T>{v1[0] + v2[0], v1[1] + v2[1]};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec2<T> operator+(const Vec2<T>& v, const T& t)
    {
        return Vec2<T>{v[0] + t, v[1] + t};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec2<T> operator-(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return Vec2<T>{v1[0] - v2[0], v1[1] - v2[1]};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec2<T> operator-(const Vec2<T>& v, const T& t)
    {
        return Vec2<T>{v[0] - t, v[1] - t};
    }

    template <typename T> 
    INLINE HOSTDEVICE Vec2<T> operator*(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return Vec2<T>{v1[0] * v2[0], v1[1] * v2[1]};
    }

    template <typename T> 
    INLINE HOSTDEVICE Vec2<T> operator*(const float t, const Vec2<T>& v)
    {
        return Vec2<T>{v[0] * t, v[1] * t};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec2<T> operator*(const Vec2<T>& v, const float t)
    {
        return t * v;
    }

    template <typename T> 
    INLINE HOSTDEVICE Vec2<T> operator/(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return Vec2<T>{v1[0] / v2[0], v1[1] / v2[1]};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec2<T> operator/(const Vec2<T>& v, const float t)
    {
        return v * (1 / t);
    }

    template <typename T> 
    INLINE HOSTDEVICE T dot(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return v1[0] * v2[0] + v1[1] * v2[1];
    }

    template <typename T> 
    INLINE HOSTDEVICE T cross(const Vec2<T>& v1, const Vec2<T>& v2)
    {
        return v1[0] * v2[1] - v1[1] * v2[0];
    }

    template <typename T>
    INLINE HOSTDEVICE T length(const Vec2<T>& v)
    {
        return sqrtf(lengthSquared(v));
    }

    template <typename T>
    INLINE HOSTDEVICE T lengthSquared(const Vec2<T>& v)
    {
        return dot(v, v);
    }

    template <typename T> 
    INLINE HOSTDEVICE Vec2<T> normalize(const Vec2<T>& v)
    {
        return v / length(v);
    }

    template <typename T>
    INLINE HOSTDEVICE Vec2<T> clamp(const Vec2<T>& v, const T a, const T b)
    {
        return Vec2<T>{clamp(v[0], a, b), clamp(v[0], a, b)};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec2<T> faceforward(const Vec2<T>& n, const Vec2<T>& i, const Vec2<T>& nref)
    {
        return n * copysignf(1.0f, dot(i, nref));
    }

    // ----------------------------------------------------------------------
    // Utilities for Vec3
    // ----------------------------------------------------------------------
#ifndef __CUDACC__
    template <typename T>
    INLINE std::ostream& operator<<(std::ostream& out, const Vec3<T>& v)
    {
        return out << v.x() << ' ' << v.y() << ' ' << v.z();
    }
#endif

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> operator+(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>{v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> operator+(const Vec3<T>& v, const T& t)
    {
        return Vec2<T>{v[0] - t, v[1] - t};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> operator-(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>{v1[0] - v2[0], v1[1] - v2[1], v1[1] - v2[2]};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> operator-(const Vec3<T>& v, const T& t)
    {
        return Vec2<T>{v[0] - t, v[1] - t};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> operator*(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>{v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> operator*(const float t, const Vec3<T>& v)
    {
        return Vec3<T>{v[0] * t, v[1] * t, v[2] * t};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> operator*(const Vec3<T>& v, const float t)
    {
        return t * v;
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> operator/(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>{v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2]};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> operator/(const Vec3<T>& v, const float t)
    {
        return v * (1 / t);
    }

    template <typename T>
    INLINE HOSTDEVICE T dot(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> cross(const Vec3<T>& v1, const Vec3<T>& v2)
    {
        return Vec3<T>{
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0]
        };
    }

    template <typename T>
    INLINE HOSTDEVICE T length(const Vec3<T>& v)
    {
        return sqrtf(lengthSquared(v));
    }

    template <typename T>
    INLINE HOSTDEVICE T lengthSquared(const Vec3<T>& v)
    {
        return dot(v, v);
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> normalize(const Vec3<T>& v)
    {
        return v / length(v);
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> clamp(const Vec3<T>& v, const T a, const T b)
    {
        return Vec3<T>{clamp(v[0], a, b), clamp(v[0], a, b), clamp(v[0], a, b)};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3<T> faceforward(const Vec3<T>& n, const Vec3<T>& i, const Vec3<T>& nref)
    {
        return n * copysignf(1.0f, dot(i, nref));
    }

    // ----------------------------------------------------------------------
    // Utilities for Vec4
    // ----------------------------------------------------------------------
#ifndef __CUDACC__
    template <typename T>
    INLINE std::ostream& operator<<(std::ostream& out, const Vec4<T>& v)
    {
        return out << v.x() << ' ' << v.y() << ' ' << v.z() << ' ' << v.w();
    }
#endif

    template <typename T>
    INLINE HOSTDEVICE Vec4<T> operator+(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>{v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2], v1[3] + v2[3]};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec4<T> operator+(const Vec4<T>& v, const T& t)
    {
        return Vec4<T>(v[0] + t, v[1] + t, v[2] + t, v[3] + t);
    }

    template <typename T>
    INLINE HOSTDEVICE Vec4<T> operator-(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>{v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2], v1[3] - v2[3]};
    }
    
    template <typename T>
    INLINE HOSTDEVICE Vec4<T> operator-(const Vec4<T>& v, const T& t)
    {
        return Vec4<T>(v[0] - t, v[1] - t, v[2] - t, v[3] - t);
    }

    template <typename T>
    INLINE HOSTDEVICE Vec4<T> operator*(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>{v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2], v1[3] * v2[3]};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec4<T> operator*(const float t, const Vec4<T>& v)
    {
        return Vec4<T>{v[0] * t, v[1] * t, v[2] * t, v[3] * t};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec4<T> operator*(const Vec4<T>& v, const float t)
    {
        return t * v;
    }

    template <typename T>
    INLINE HOSTDEVICE Vec4<T> operator/(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>{v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2], v1[3] / v2[3]};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec4<T> operator/(const Vec4<T>& v, const float t)
    {
        return v * (1 / t);
    }

    template <typename T>
    INLINE HOSTDEVICE T dot(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    }

    template <typename T>
    INLINE HOSTDEVICE Vec4<T> cross(const Vec4<T>& v1, const Vec4<T>& v2)
    {
        return Vec4<T>{
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0], 
            1.0f
        };
    }

    template <typename T>
    INLINE HOSTDEVICE T length(const Vec4<T>& v)
    {
        return sqrtf(lengthSquared(v));
    }

    template <typename T>
    INLINE HOSTDEVICE T lengthSquared(const Vec4<T>& v)
    {
        return dot(v, v);
    }

    template <typename T>
    INLINE HOSTDEVICE Vec4<T> normalize(const Vec4<T>& v)
    {
        return v / length(v);
    }

    template <typename T>
    INLINE HOSTDEVICE Vec4<T> clamp(const Vec4<T>& v, const float a, const float b)
    {
        return Vec4<T>{clamp(v[0], a, b), clamp(v[0], a, b), clamp(v[0], a, b), clamp(v[0], a, b)};
    }

    template <typename T>
    INLINE HOSTDEVICE Vec4<T> faceforward(const Vec4<T>& n, const Vec4<T>& i, const Vec4<T>& nref)
    {
        return n * copysignf(1.0f, dot(i, nref));
    }

    VEC_DECL_T(Vec2)
    VEC_DECL_T(Vec3)
    VEC_DECL_T(Vec4)
}