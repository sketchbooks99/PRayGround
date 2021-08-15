#pragma once

#include <sutil/vec_math.h>

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

template <typename T, unsigned int N> bool          operator==(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> bool          operator!=(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> Matrix<T, N>  operator+(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> Matrix<T, N>& operator+=(Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> Matrix<T, N>  operator-(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> Matrix<T, N>& operator-=(Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> Matrix<T, N>  operator*(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> Matrix<T, N>  operator*(const Matrix<T, N>& m, const float t);
template <typename T, unsigned int N> Matrix<T, N>& operator*=(Matrix<T, N>& m1, const Matrix<T, N>& m2);
template <typename T, unsigned int N> Matrix<T, N>& operator*=(Matrix<T, N>& m, const float t);
template <typename T, unsigned int N> Matrix<T, N>  operator/(const Matrix<T, N>& m1, const float t);
template <typename T, unsigned int N> typename Matrix<T, N>::floatN operator*(const Matrix<T, N>& m, typename Matrix<T, N>::floatN& v);

template <typename T> float4 operator*(const Matrix<T, 3>, const float4& v);
template <typename T> float3 operator*(const Matrix<T, 4>, const float3& v);

 
// Class definition
template <typename T, unsigned int N>
class Matrix
{
public:
    using floatN = typename Vector<T, N>::Type;
    using TfloatN = typename Vector<T, N>::TransformType;

    explicit HOSTDEVICE Matrix();
    explicit HOSTDEVICE Matrix(const Matrix& m);
    explicit HOSTDEVICE Matrix(const T data[N*N]);
    explicit HOSTDEVICE Matrix(const float data[12]);
    explicit HOSTDEVICE Matrix(const std::initializer_list<T>& list);

    HOSTDEVICE T  operator[](int i) const;
    HOSTDEVICE T& operator[](int i);

    HOSTDEVICE Matrix& operator=( const Matrix& a );

    HOSTDEVICE void setColumn(const floatN& v, int col_idx);
    HOSTDEVICE void setRow(const floatN& v, int row_idx);

    HOSTDEVICE T  get(int i, int j) const;
    HOSTDEVICE T& get(int i, int j);

    HOSTDEVICE void setData(const T data[N*N]);

    HOSTDEVICE       T* data();
    HOSTDEVICE const T* data() const;

    HOSTDEVICE bool isIdentity() const;

    /**
     * @note
     * [JP] 以下の変換行列関数は \c N に応じて特殊化されます。
     * [EN] The following transformation functions specilized depending on \c N.
     */
    static HOSTDEVICE Matrix rotate(const float radians, const TfloatN& axis = {});
    static HOSTDEVICE Matrix translate(const TfloatN& t);
    static HOSTDEVICE Matrix scale(const TfloatN& s);
    static HOSTDEVICE Matrix scale(const float s);

    static HOSTDEVICE Matrix zero();
    static HOSTDEVICE Matrix identity();
private:
    T m_data[N*N];
};

// Operator overload
// ----------------------------------------------------------------------------
template <typename T, unsigned int N> 
bool operator==(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    bool is_equal = true;
    for (int i = 0; i < N*N; i++)
        is_equal &= (m1[i] == m2[i]);
    return is_equal;
}

template <typaname T, unsigned int N>
bool operator!=(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    return !(m1 == m2);
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
Matrix<T, N> operator+(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    Matrix<T, N> result;
    for (int i = 0; i < N*N; i++)
        result[i] = m1[i] + m2[i];
    return result;
}

template <typename T, unsigned int N>
Matrix<T, N>& operator+=(Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    for (int i = 0; i < N*N; i++)
        m1[i] += m2[i];
}

template <typename T, unsigned int N>
Matrix<T, N> operator-(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    Matrix<T, N> result;
    for (int i = 0; i < N*N; i++)
        result[i] = m1[i] - m2[i];
    return result;
}

template <typename T, unsigned int N>
Matrix<T, N>& operator-=(Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    for (int i = 0; i < N*N; i++)
        m1[idx] -= m2[idx];
}

template <typename T, unsigned int N>
Matrix<T, N> operator*(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    Matrix<T, N> result = Matrix<T, N>::zero();
    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            for (int tmp = 0; tmp < N; tmp++)
            {
                int i = row * N + col;
                int m1_i = row * N + tmp;
                int m2_i = tmp * N + col;
                result[i] += m1[m1_i] * m2[m2_i];
            }
        }
    }
    return result;
}

template <typename T, unsigned int N>
Matrix<T, N> operator*(const Matrix<T, N>& m, const float t)
{
    Matrix<T, N> result;
    for (int i = 0; i < N*N; i++)
        result[i] = m[i] * t;
    return result;
}

template <typename T, unsigned int N>
Matrix<T, N>& operator*=(Matrix<T, N>& m1, const Matrix<T, N>& m2)
{
    for (int i = 0; i < N*N; i++)
        m1[i] *= m2[i];
}

template <typename T, unsigned int N>
Matrix<T, N>& operator*=(Matrix<T, N>& m1, const float t)
{
    for (int i = 0; i < N*N; i++)
        m1[i] *= t;
}

template <typename T, unsigned int N>
Matrix<T, N> operator/(const Matrix<T, N>& m1, const float t)
{
    Matrix<T, N> result;
    for (int i = 0; i < N*N; i++)
        result[i] = m1[i] / t;
    return result;
}

template <typename T, unsigned int N>
typename Vector<T, N>::Type operator*(const Matrix<T, N>& m, const Matrix<T, N>::floatN& v)
{
    using vec_t = typename Vector<T, N>::Type;
    vec_t result;
    T* result_ptr = reinterpret_cast<T*>(&result);
    const T* v_ptr = reinterpret_cast<const T*>(&v);

    for (int row = 0; row < N; row++)
    {
        T sum = static_cast<T>(0);
        for (int col = 0; col < N; col++)
        {
            sum += m[row * N + col] * v_ptr[col];
        }
        result[row] = sum;
    }
    return result;
}

template <typename T>
float4 operator*(const Matrix<T, 3>& m, const float4& v)
{
    float3 tmp = make_float3(v.x, v.y, v.z);
    tmp = m * tmp;
    return make_float4(tmp.x, tmp.y, tmp.z, 1.0f);
}

template <typename T>
float3 operator*(const Matrix<T, 4>& m, const float3& v)
{
    float4 tmp = make_float4(v.x, v.y, v.z, 1.0f);
    tmp = m * tmp;
    return make_float3(tmp.x, tmp.y, tmp.z);
}


// Class implementations
// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N>::Matrix() : Matrix(Matrix::identity()) { }

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
HOSTDEVICE Matrix<T, N>::Matrix(const float data[12])
{
    static_assert(std::is_same_v<T, float> && N == 4
        "This constructor is only allowed for Matrix4f");

    for (int i = 0; i < 12; i++)
        m_data[i] = data[i];
    m_data[12] = m_data[13] = m_data[14] = 0.0f;
    m_data[15] = 1.0f;
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
HOSTDEVICE void Matrix<T, N>::setData(const T data[N*N])
{
    for (int i = 0; i < N*N; i++)
        m_data[i] = data[i];
}

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
template <typename T, unsigned int N>
HOSTDEVICE bool Matrix<T, N>::isIdentity() const
{
    return *this == Matrix<T, N>::identity();
}

// ----------------------------------------------------------------------------
template <typaneme T, unsigned int N>
HOSTDEVICE Matrix<T, N> Matrix<T, N>::rotate(const float radians, const typename Matrix<T, N>::TfloatN& axis)
{
    static_assert(1 < N && N <= 4, "Matrix::rotate(): The dimension of matrix must be 2x2, 3x3 or 4x4.");

    const float s = sinf(radians);
    const float c = cosf(radians);

    Matrix<T, N> result{};

    if constexpr (N == 2)
    {
        const T data[N*N] = { c, -s, s, c };
        result.setData(data);
    }
    else if constexpr (N == 3 || N == 4)
    {
        float3 a = normalize(axis);
        // 1st row
        result[0] = a.x * a.x + (1.0 - a.x * a.x) * c;
        result[1] = a.x * a.y + (1.0 - c) - a.z * s;
        result[2] = a.x * a.z + (1.0 - c) + a.y * s;

        // 2nd row
        result[1 * N + 0] = a.x * a.y * (1.0 - c) + a.z * s;
        result[1 * N + 1] = a.y * a.y + (1.0 - a.y * a.y) * c;
        result[1 * N + 2] = a.y * a.z * (1.0 - c) - a.x * s;

        // 3rd row
        result[2 * N + 0] = a.x * a.z * (1.0 - c) - a.y * s;  
        result[2 * N + 1] = a.y * a.z * (1.0 - c) - a.y * s;
        result[2 * N + 2] = a.z * a.z + (1.0 - a.z * a.z) * c;
    }
    return result;
}

template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N> Matrix<T, N>::translate(const typename Matrix<T, N>::TfloatN& t)
{
    static_assert(1 < N && N <= 4, "Matrix::translate(): The dimension of matrix must be 2x2, 3x3 or 4x4.");

    Matrix<T, N> result{};
    const float* t_ptr = reinterpret_cast<const float*>(&t);
    for (size_t row = 0; row < static_cast<size_t>((sizeof(t) / sizeof(float))); row++)
        result[row * N + (N-1)] = t_ptr[row];
    return result;
}

template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N> Matrix<T, N>::scale(const typename Matrix<T, N>::TfloatN& s)
{
    static_assert(1 < N && N <= 4, "Matrix::scale(): The dimension of matrix must be 2x2, 3x3 or 4x4.");

    Matrix<T, N> result{};
    const float* s_ptr = reinterpret_cast<const float*>(&s);
    for (size_t i = 0; i < static_cast<size_t>((sizeof(s) / sizeof(float)); i++)
        result[i * N + i] = s_ptr[i];
    return result;
}

template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N> Matrix<T, N>::scale(const float s)
{
    static_assert(1 < N && N <= 4, "Matrix::scale(): The dimension of matrix must be 2x2, 3x3 or 4x4.");

    Matrix<T, N> result{};
    for (size_t i = 0; i < static_cast<size_t>(sizeof(Matrix<T, N>::TfloatN) / sizeof(float)); i++)
        result[i * N + i] = s;
    return result;
}

// ----------------------------------------------------------------------------
template <typename T, unsigned int N>
HOSTDEVICE Matrix<T, N> Matrix<T, N>::zero()
{
    T data[N*N] = {};
    return Matrix(data);
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
    return Matrix(data);
}

}
