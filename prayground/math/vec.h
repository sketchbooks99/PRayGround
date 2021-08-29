/**
 * @file vec.h
 * @author Shunji Kiuchi (lunaearth445@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-08-23
 * 
 * @copyright Copyright (c) 2021
 * 
 * @todo 
 * ベクトルライブラリの実装
 * もしかしたらデバイス側でも Vector4f 等を使えるようにするかも
 */

#pragma once

#include <vector_functions.h>
#include <vector_types.h>
#include <prayground/optix/macros.h>

#ifndef __CUDACC__
#include <cmath>
#include <cstdlib>
#endif

namespace prayground {

namespace {

// float4 -> Vector4f のような代入なり初期化ができるようにするためのダミーの構造体
// 無名名前空間に入れといて外部からは見えないようにする
template <typename T, uint32_t N> struct CUVec { using Type = T; };
template <> struct CUVec<float, 2> { using Type = float2; };
template <> struct CUVec<float, 3> { using Type = float3; };
template <> struct CUVec<float, 4> { using Type = float4; };
template <> struct CUVec<int, 2>   { using Type = int2; };
template <> struct CUVec<int, 3>   { using Type = int3; };
template <> struct CUVec<int, 4>   { using Type = int4; };

} // nonamed namespace

// forward declarations
template <typename T, uint32_t N> class Vector;
using Vector2f = Vector<float, 2>;
using Vector3f = Vector<float, 3>;
using Vector4f = Vector<float, 4>;
using Vector2i = Vector<int, 2>;
using Vector3i = Vector<int, 3>;
using Vector4i = Vector<int, 4>;

template <typename T, uint32_t N> INLINE HOSTDEVICE bool          operator==(const Vector<T, N>& v1, const Vector<T, N>& v2);
template <typename T, uint32_t N> INLINE HOSTDEVICE bool          operator==(const Vector<T, N>& v1, const Vector<T, N>& v2);
template <typename T, uint32_t N> INLINE HOSTDEVICE Vector<T, N>  operator+(const Vector<T, N>& v1, const Vector<T, N>& v2);
template <typename T, uint32_t N> INLINE HOSTDEVICE Vector<T, N>& operator+=(Vector<T, N>& v1, const Vector<T, N>& v2);
template <typename T, uint32_t N> INLINE HOSTDEVICE Vector<T, N>  operator-(const Vector<T, N>& v1, const Vector<T, N>& v2);
template <typename T, uint32_t N> INLINE HOSTDEVICE Vector<T, N>& operator-=(Vector<T, N>& v1, const Vector<T, N>& v2);
template <typename T, uint32_t N> INLINE HOSTDEVICE Vector<T, N>  operator*(const Vector<T, N>& v1, const Vector<T, N>& v2);
template <typename T, uint32_t N> INLINE HOSTDEVICE Vector<T, N>& operator*=(Vector<T, N>& v1, const Vector<T, N>& v2);

template <typename T, uint32_t N>
class Vector 
{
public:
    using CUType = typename CUVec<T, N>::Type;

    HOSTDEVICE Vector();
    HOSTDEVICE Vector(const std::initializer_list<T>& list);
    HOSTDEVICE Vector(const T[N] data);
    HOSTDEVICE Vector(const Vector& v);
    HOSTDEVICE Vector(const CUType& cu_v);

    // (float4)Vector4f ができるようにする
    // (int4)Vector4f -> NG
    HOSTDEVICE operator CUType() const;

    HOSTDEVICE Vector& operator=(const Vector& v);

    HOSTDEVICE T  operator[](uint32_t i) const;
    HOSTDEVICE T& operator[](uint32_t i);

    HOSTDEVICE       T* data();
    HOSTDEVICE const T* data() const;

private:
    T[N] m_data;
};

} // ::prayground