#pragma once

#include <prayground/math/vec_math.h>
#include <prayground/math/util.h>

#ifndef __CUDACC__
    #include <iostream>
#endif

namespace prayground {

// Forward declarations
template <typename T, unsigned int N> class Matrix;
using Matrix4f = Matrix<float, 4>;
using Matrix3f = Matrix<float, 3>;
using Matrix2f = Matrix<float, 2>;

template <typename T, unsigned int N> struct Vector { };
template <> struct Vector<float, 2> { using Type = float2; using TransformType = float2; };
template <> struct Vector<float, 3> { using Type = float3; using TransformType = float3; };
template <> struct Vector<float, 4> { using Type = float4; using TransformType = float3; };

template <typename T, unsigned int N> INLINE HOSTDEVICE bool          operator==(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> INLINE HOSTDEVICE bool          operator!=(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> INLINE HOSTDEVICE Matrix<T, N>  operator+(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> INLINE HOSTDEVICE Matrix<T, N>& operator+=(Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> INLINE HOSTDEVICE Matrix<T, N>  operator-(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> INLINE HOSTDEVICE Matrix<T, N>& operator-=(Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> INLINE HOSTDEVICE Matrix<T, N>  operator*(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> INLINE HOSTDEVICE Matrix<T, N>  operator*(const Matrix<T, N>& m, const float t);
template <typename T, unsigned int N> INLINE HOSTDEVICE Matrix<T, N>& operator*=(Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> INLINE HOSTDEVICE Matrix<T, N>& operator*=(Matrix<T, N>& m, const float t);
template <typename T, unsigned int N> INLINE HOSTDEVICE Matrix<T, N>  operator/(const Matrix<T, N>& m1, const float t);
template <typename T, unsigned int N> INLINE HOSTDEVICE typename Matrix<T, N>::floatN operator*(const Matrix<T, N>& m, typename Matrix<T, N>::floatN& v);

template <typename T> INLINE HOSTDEVICE float4 operator*(const Matrix<T, 3>, const float4& v);
template <typename T> INLINE HOSTDEVICE float3 operator*(const Matrix<T, 4>, const float3& v);
 
// Class definition
template <typename T, unsigned int N>
class Matrix
{
public:
    using floatN = typename Vector<T, N>::Type;
    using TfloatN = typename Vector<T, N>::TransformType;

    HOSTDEVICE Matrix();
    HOSTDEVICE Matrix(const Matrix& m);
    HOSTDEVICE Matrix(const T data[N*N]);
    HOSTDEVICE Matrix(const float (&data)[12]);
    HOSTDEVICE Matrix(const std::initializer_list<T>& list);

    HOSTDEVICE const T& operator[](unsigned int i) const;
    HOSTDEVICE       T& operator[](unsigned int i);

    HOSTDEVICE Matrix& operator=( const Matrix& a );

    HOSTDEVICE void setColumn(const floatN& v, unsigned int col_idx);
    HOSTDEVICE void setRow(const floatN& v, unsigned int row_idx);

    HOSTDEVICE const T& get(unsigned int i, unsigned int j) const;
    HOSTDEVICE       T& get(unsigned int i, unsigned int j);

    HOSTDEVICE void setData(const T data[N*N]);

    HOSTDEVICE       T* data();
    HOSTDEVICE const T* data() const;

    HOSTDEVICE bool isIdentity() const;

    HOSTDEVICE Matrix inverse() const;

    HOSTDEVICE float3 pointMul(const float3& p) const;
    HOSTDEVICE float3 vectorMul(const float3& v) const;
    HOSTDEVICE float3 normalMul(const float3& n) const;

    static HOSTDEVICE Matrix           rotate(const float radians, const TfloatN& axis);
    static HOSTDEVICE Matrix<float, 4> translate(const float3& t);
    static HOSTDEVICE Matrix           scale(const TfloatN& s);
    static HOSTDEVICE Matrix           scale(const float s);
    static HOSTDEVICE Matrix<float, 4> shear(float a, float b, Axis axis);
    static HOSTDEVICE Matrix<float, 4> reflection(Axis axis);

    static HOSTDEVICE Matrix zero();
    static HOSTDEVICE Matrix identity();
private:
    T m_data[N*N];
};

#ifndef __CUDACC__
template <typename T, unsigned int N>
inline std::ostream& operator<<(std::ostream& out, Matrix<T, N> m)
{
    for (int i = 0; i < N*N; i++)
        out << m[i] << ' ';
    return out;
}
#endif // __CUDACC__

// Operator overload
// ----------------------------------------------------------------------------
template <typename T, unsigned int N> 
INLINE HOSTDEVICE bool operator==(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    bool is_equal = true;
    for (unsigned int i = 0; i < N*N; i++)
        is_equal &= (m1[i] == m2[i]);
    return is_equal;
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE bool operator!=(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    return !(m1 == m2);
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N> operator+(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    Matrix<T, N> result(m1);
    result += m2;
    return result;
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N>& operator+=(Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    for (unsigned int i = 0; i < N*N; i++)
        m1[i] += m2[i];
    return m1;
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N> operator-(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    Matrix<T, N> result(m1);
    result -= m2;
    return result;
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N>& operator-=(Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    for (unsigned int i = 0; i < N*N; i++)
        m1[i] -= m2[i];
    return m1;
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N> operator*(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    T* data = new T[N * N];
    for (unsigned int row = 0; row < N; row++)
    {
        for (unsigned int col = 0; col < N; col++)
        {
            T sum = static_cast<T>(0);
            for (unsigned int tmp = 0; tmp < N; tmp++)
            {
                unsigned int m1_i = row * N + tmp;
                unsigned int m2_i = tmp * N + col;
                sum += m1[m1_i] * m2[m2_i];
            }
            data[row * N + col] = sum;
        }
    }
    return Matrix<T, N>(data);
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N> operator*(const Matrix<T, N>& m, const float t)
{
    Matrix<T, N> result;
    for (unsigned int i = 0; i < N*N; i++)
        result[i] = m[i] * t;
    return result;
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N>& operator*=(Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    for (unsigned int i = 0; i < N*N; i++)
        m1[i] *= m2[i];
    return m1;
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N>& operator*=(Matrix<T, N>& m1, const float t)
{
    for (unsigned int i = 0; i < N*N; i++)
        m1[i] *= t;
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N> operator/(const Matrix<T, N>& m1, const float t)
{
    Matrix<T, N> result;
    for (unsigned int i = 0; i < N*N; i++)
        result[i] = m1[i] / t;
    return result;
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE typename Vector<T, N>::Type operator*(const Matrix<T, N>& m, const typename Matrix<T, N>::floatN& v)
{
    using vec_t = typename Vector<T, N>::Type;
    vec_t result;
    T* result_ptr = reinterpret_cast<T*>(&result);
    const T* v_ptr = reinterpret_cast<const T*>(&v);

    for (unsigned int row = 0; row < N; row++)
    {
        T sum = static_cast<T>(0);
        for (unsigned int col = 0; col < N; col++)
        {
            sum += m[row * N + col] * v_ptr[col];
        }
        result[row] = sum;
    }
    return result;
}

template <typename T>
INLINE HOSTDEVICE float4 operator*(const Matrix<T, 3>& m, const float4& v)
{
    float3 tmp = make_float3(v.x, v.y, v.z);
    tmp = m * tmp;
    return make_float4(tmp.x, tmp.y, tmp.z, 1.0f);
}

template <typename T>
INLINE HOSTDEVICE float3 operator*(const Matrix<T, 4>& m, const float3& v)
{
    float4 tmp = make_float4(v.x, v.y, v.z, 1.0f);
    tmp = m * tmp;
    return make_float3(tmp.x, tmp.y, tmp.z);
}


// Class implementations
// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N>::Matrix() : Matrix(Matrix::identity()) { }

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N>::Matrix(const Matrix<T, N>& m)
{
    for (unsigned int i = 0; i < N*N; i++)
        m_data[i] = m[i];
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N>::Matrix(const T data[N*N])
{
    for (unsigned int i = 0; i < N * N; i++)
        m_data[i] = data[i];
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N>::Matrix(const float (&data)[12])
{
    static_assert(std::is_same_v<T, float> && N == 4,
        "This constructor is only allowed for Matrix4f");

    for (unsigned int i = 0; i < 12; i++)
        m_data[i] = data[i];
    m_data[12] = m_data[13] = m_data[14] = 0.0f;
    m_data[15] = 1.0f;
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N>::Matrix(const std::initializer_list<T>& list)
{
    unsigned int i = 0; 
    for (auto it = list.begin(); it != list.end(); ++it)
        m_data[ i++ ] = *it;
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N>& Matrix<T, N>::operator=(const Matrix<T, N>& a)
{
    for (unsigned int i = 0; i < N * N; i++)
        m_data[i] = a[i];
    return *this;
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
INLINE HOSTDEVICE const T& Matrix<T, N>::operator[](const unsigned int i) const
{
    return m_data[i];
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE T& Matrix<T, N>::operator[](const unsigned int i)
{
    return m_data[i];
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
INLINE HOSTDEVICE const T& Matrix<T, N>::get(unsigned int i, unsigned int j) const
{
    unsigned int idx = i * N + j;
    return m_data[idx];
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE T& Matrix<T, N>::get(unsigned int i, unsigned int j)
{
    unsigned int idx = i * N + j;
    return m_data[idx];
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
INLINE HOSTDEVICE void Matrix<T, N>::setData(const T data[N*N])
{
    for (unsigned int i = 0; i < N*N; i++)
        m_data[i] = data[i];
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE T* Matrix<T, N>::data()
{
    return m_data;
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE const T* Matrix<T, N>::data() const
{
    return m_data;
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
INLINE HOSTDEVICE bool Matrix<T, N>::isIdentity() const
{
    return *this == Matrix<T, N>::identity();
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N> Matrix<T, N>::inverse() const
{
    Matrix<T, N> i_mat = Matrix<T, N>::identity();
    T* inv_data = i_mat.data();
    Matrix<T, N> mat(*this);
    float tmp;
    for (unsigned int i = 0; i < N; i++)
    {
        tmp = 1.0f / mat[i * N + i];
        for (unsigned int j = 0; j < N; j++)
        {
            mat[i * N + j] *= tmp;
            inv_data[i * N + j] *= tmp;
        }
        for (unsigned int j = 0; j < N; j++)
        {
            if (i != j)
            {
                tmp = mat[j * N + i];
                for (unsigned int k = 0; k < N; k++)
                {
                    mat[j * N + k] -= mat[i * N + k] * tmp;
                    inv_data[j * N + k] -= inv_data[i * N + k] * tmp;
                }
            }
        }
    }
    return Matrix<T, N>(inv_data);
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
INLINE HOSTDEVICE float3 Matrix<T, N>::pointMul(const float3& p) const 
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<N, 4>);
    return make_float3(0.0f);
}

template<>
INLINE HOSTDEVICE float3 Matrix<float, 4>::pointMul(const float3& p) const
{
    float x = m_data[0]  * p.x + m_data[1]  * p.x + m_data[2]  * p.z + m_data[3];
    float y = m_data[4]  * p.x + m_data[5]  * p.x + m_data[6]  * p.z + m_data[7];
    float z = m_data[8]  * p.x + m_data[9]  * p.x + m_data[10] * p.z + m_data[11];
    float w = m_data[12] * p.x + m_data[13] * p.x + m_data[14] * p.z + m_data[15];

    // w == 0のときのAssertionは必要
    if (w == 1.0f)
        return make_float3(x, y, z);
    else
        return make_float3(x, y, z) / w;
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE float3 Matrix<T, N>::vectorMul(const float3& n) const 
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<N, 4>);
    return make_float3(0.0f);
}

template<>
INLINE HOSTDEVICE float3 Matrix<float, 4>::vectorMul(const float3& v) const
{
    float x = v.x, y = v.y, z = v.z;
    return make_float3(
        m_data[0] * x + m_data[1] * y + m_data[2]  * z,        
        m_data[4] * x + m_data[5] * y + m_data[6]  * z,
        m_data[8] * x + m_data[9] * y + m_data[10] * z
    );
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE float3 Matrix<T, N>::normalMul(const float3& n) const 
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<N, 4>);
    return make_float3(0.0f);
}

template<>
INLINE HOSTDEVICE float3 Matrix<float, 4>::normalMul(const float3& n) const
{   
    Matrix4f inv = this->inverse();
    const float* inv_data = inv.data();

    float x = n.x, y = n.y, z = n.z;
    return make_float3(
        inv_data[0] * x + inv_data[4] * y + inv_data[8]  * z,
        inv_data[1] * x + inv_data[5] * y + inv_data[9]  * z,
        inv_data[2] * x + inv_data[6] * y + inv_data[10] * z
    );
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N> Matrix<T, N>::rotate(const float radians, const typename Matrix<T, N>::TfloatN& axis)
{
    static_assert(1 < N && N <= 4, "Matrix::rotate(): The dimension of matrix must be 2x2, 3x3 or 4x4.");

    const float s = sinf(radians);
    const float c = cosf(radians);

    Matrix<T, N> i_mat = Matrix<T, N>::identity();
    T* data = i_mat.data();

    if constexpr (N == 2)
    {
        data[0] = c; data[1] = -s; 
        data[2] = s; data[3] = c;
        return Matrix<T, N>(data);
    }
    else if constexpr (N == 3 || N == 4)
    {
        float3 a = normalize(axis);
        // 1st row
        data[0] = a.x * a.x + (1.0 - a.x * a.x) * c;
        data[1] = a.x * a.y * (1.0 - c) - a.z * s;
        data[2] = a.x * a.z * (1.0 - c) + a.y * s;

        // 2nd row
        data[1 * N + 0] = a.x * a.y * (1.0 - c) + a.z * s;
        data[1 * N + 1] = a.y * a.y + (1.0 - a.y * a.y) * c;
        data[1 * N + 2] = a.y * a.z * (1.0 - c) - a.x * s;

        // 3rd row
        data[2 * N + 0] = a.x * a.z * (1.0 - c) - a.y * s;  
        data[2 * N + 1] = a.y * a.z * (1.0 - c) + a.x * s;
        data[2 * N + 2] = a.z * a.z + (1.0 - a.z * a.z) * c;
    }
    return Matrix<T, N>(data);
}

template <>
INLINE HOSTDEVICE Matrix<float, 4> Matrix<float, 4>::translate(const float3 &t)
{
    Matrix4f i_mat = Matrix4f::identity();
    float* data = i_mat.data();
    data[0 * 4 + 3] = t.x;
    data[1 * 4 + 3] = t.y;
    data[2 * 4 + 3] = t.z;
    return Matrix4f(data);
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N> Matrix<T, N>::scale(const typename Matrix<T, N>::TfloatN& s)
{
    static_assert(1 < N && N <= 4, "Matrix::scale(): The dimension of matrix must be 2x2, 3x3 or 4x4.");

    Matrix<T, N> i_mat = Matrix<T, N>::identity();
    float* data = i_mat.data();
    const float* s_ptr = reinterpret_cast<const float*>(&s);
    for (size_t i = 0; i < static_cast<size_t>(sizeof(s) / sizeof(float)); i++)
        data[i * N + i] = s_ptr[i];
    return Matrix<T, N>(data);
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N> Matrix<T, N>::scale(const float s)
{
    static_assert(1 < N && N <= 4, "Matrix::scale(): The dimension of matrix must be 2x2, 3x3 or 4x4.");

    Matrix<T, N> i_mat = Matrix<T, N>::identity();
    float* data = i_mat.data();
    for (size_t i = 0; i < static_cast<size_t>(sizeof(Matrix<T, N>::TfloatN) / sizeof(float)); i++)
        data[i * N + i] = s;
    return Matrix<T, N>(data);
}

template <>
INLINE HOSTDEVICE Matrix<float, 4> Matrix<float, 4>::reflection(Axis axis)
{
    Matrix4f i_mat = Matrix4f::identity();
    float* data = i_mat.data();
    unsigned int i = static_cast<unsigned int>(axis);
    data[i * 4 + i] *= -1;
    return Matrix4f(data);
}

template <>
INLINE HOSTDEVICE Matrix<float, 4> Matrix<float, 4>::shear(float a, float b, Axis axis)
{
    Matrix<float, 4> i_mat = Matrix<float, 4>::identity();
    float* data = i_mat.data();
    switch(axis)
    {
        case Axis::X:
            data[4] = a; data[8] = b;
            return Matrix4f(data);
        case Axis::Y:
            data[1] = a; data[9] = b;
            return Matrix4f(data);
        case Axis::Z:
            data[2] = a; data[6] = b;
            return Matrix4f(data);        
    }
    return Matrix4f(data);
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N> Matrix<T, N>::zero()
{
    T data[N*N];
    for (unsigned int i = 0; i < N*N; i++)
        data[i] = 0;
    return Matrix<T, N>(data);
}

template <typename T, unsigned int N>
INLINE HOSTDEVICE Matrix<T, N> Matrix<T, N>::identity()
{
    T data[N*N];
    for (unsigned int i = 0; i < N; i++)
    {
        for (unsigned int j = 0; j < N; j++)
        {
            data[i * N + j] = i == j ? static_cast<T>(1) : static_cast<T>(0);
        }
    }
    return Matrix<T, N>(data);
}

} // ::prayground