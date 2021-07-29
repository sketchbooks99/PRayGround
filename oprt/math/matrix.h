#pragma once

#include "vec.h"

namespace oprt {

// Forward declarations
template <typename T, unsigned int N> class Matrix;
using Matrix4f = Matrix<float, 4>;
using Matrix3f = Matrix<float, 3>;
using Matrix2f = Matrix<float, 2>;

template <typename T, unsigned int N> struct Vector { };
template <> struct Vector<float, 2> { using Type = float2; using TransformType = float2; };
template <> struct Vector<float, 3> { using Type = float3; using TransformType = float3; };
template <> struct Vector<float, 4> { using Type = float4; using TransformType = float3; };

template <typename T, unsigned int N> bool          operator==(const Matrix<T, N>&, const Matrix<T, N>&);
template <typename T, unsigned int N> bool          operator!=(const Matrix<T, N>&, const Matrix<T, N>&);
template <typename T, unsigned int N> Matrix<T, N>  operator+(const Matrix<T, N>&, const Matrix<T, N>&);
template <typename T, unsigned int N> Matrix<T, N>& operator+=(const Matrix<T, N>&, const Matrix<T, N>&);
template <typename T, unsigned int N> Matrix<T, N>  operator-(const Matrix<T, N>&, const Matrix<T, N>&);
template <typename T, unsigned int N> Matrix<T, N>& operator-=(const Matrix<T, N>&, const Matrix<T, N>&);
template <typename T, unsigned int N> Matrix<T, N>  operator*(const Matrix<T, N>&, const Matrix<T, N>&);
template <typename T, unsigned int N> Matrix<T, N>& operator*=(const Matrix<T, N>&, float);
template <typename T, unsigned int N> Matrix<T, N>& operator*=(const Matrix<T, N>&, const Matrix<T, N>&);
template <typename T, unsigned int N> Matrix<T, N>  operator/(const Matrix<T, N>&, const Matrix<T, N>&);
template <typename T, unsigned int N> typename Vector<T, N>::Type operator*(const Matrix<T, N>&, const typename Vector<T, N>::Type&);

// Class definition
template <typename T, unsigned int N>
class Matrix
{
public:
    explicit HOSTDEVICE Matrix();
    explicit HOSTDEVICE Matrix(const Matrix& m);
    explicit HOSTDEVICE Matrix(const T data[N*N]);
    explicit HOSTDEVICE Matrix(const std::initializer_list<T>& list);

    HOSTDEVICE T  operator[](int i) const;
    HOSTDEVICE T& operator[](int i);

    HOSTDEVICE Matrix& operator=( const Matrix& a );

    HOSTDEVICE T  get(int i, int j) const;
    HOSTDEVICE T& get(int i, int j);

    HOSTDEVICE       T* data();
    HOSTDEVICE const T* data() const;

    HOSTDEVICE bool isIdentity();

    /**
     * @note
     * [JP] 以下の変換行列関数は \c N に応じて特殊化されます。
     * [EN] The following transformation functions specilized depending on \c N.
     */
    static HOSTDEVICE Matrix<T, N> rotate(const float radians, const Vector<T, N>::TransformType& axis);
    static HOSTDEVICE Matrix<T, N> translate(const Vector<T, N>::TransformType& t);
    static HOSTDEVICE Matrix<T, N> scale(const Vector<T, N>::TransformType& s);
    static HOSTDEVICE Matrix<T, N> scale(const float s);

    static HOSTDEVICE Matrix<T, N> zero();
    static HOSTDEVICE Matrix<T, N> identity();
private:
    T m_data[N*N];
};

// Implementations

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N>::Matrix() : Matrix(Matrix::zero()) { }

template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N>::Matrix(const Matrix<T, N>& m)
{
    for (int i = 0; i < N*N; i++)
        m_data[i] = m[i];
}

template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N>::Matrix(const T data[N*N])
{
    for (int i = 0; i < N*N; i++)
        m_data[i] = data[i];
}

template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N>::Matrix(const std::initializer_list<T>& list)
{
    int i = 0; 
    for (auto it = list.begin(); it != list.end(); ++it)
        m_data[ i++ ] = *it;
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N>& Matrix<T, N>::operator=( const Matrix<T, N>& a )
{
    for (int i = 0; i < N*N; i++)
        m_data[i] = a[i];
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
HOSTDEVICE T Matrix<T, N>::operator[](const int i) const 
{
    return m_data[i];
}

template <typename T, unsigned int N>
HOSTDEVICE T& Matrix<T, N>::operator[](const int i)
{
    return m_data[i];
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
HOSTDEVICE T Matrix<T, N>::get(int i, int j) const
{
    unsigned int idx = i * N + j;
    return m_data[idx];
}

template <typename T, unsigned int N>
HOSTDEVICE T& Matrix<T, N>::get(int i, int j)
{
    unsigned int idx = i * N + j;
    return m_data[idx];
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
HOSTDEVICE T* Matrix<T, N>::data()
{
    return m_data;
}

template <typenmae T, unsigned int N>
HOSTDEVICE const T* Matrix<T, N>::data() const
{
    return m_data;
}

// ----------------------------------------------------------------------------
template <typaneme T, unsigned int N>
HOSTDEVICE Matrix<T, N> Matrix<T, N>::rotate(const float radians, const Vector<T, N>::TransformType& axis)
{
     
}

template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N> Matrix<T, N>::translate(const float3& t)
{

}

template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N> Matrix<T, N>::scale(const float3& s)
{
    
}

template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N> Matrix<T, N>::scale(const float s)
{
    
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N> Matrix<T, N>::zero()
{
    T zero_data[N*N] = {};
    return zero_data;
}

template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N> Matrix<T, N>::identity()
{
    T data[N*N] = {};
    for (int i=0; i<N; i++)
    {
        auto idx = i * N + i;
        data[idx] = static_cast<T>(1);
    }
    return data;
}

}
