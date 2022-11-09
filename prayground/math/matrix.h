#pragma once

#include <prayground/core/util.h>
#include <prayground/math/vec.h>
#include <prayground/math/util.h>

#ifndef __CUDACC__
    #include <iostream>
#endif

namespace prayground {

    // Forward declarations
    template <typename T, uint32_t N> class Matrix;
    using Matrix2f = Matrix<float, 2>;
    using Matrix3f = Matrix<float, 3>;
    using Matrix4f = Matrix<float, 4>;

    template <typename T, uint32_t N> struct Vector { };
    template <> struct Vector<float, 2> { using Type = Vec2f; using TransformType = Vec2f; };
    template <> struct Vector<float, 3> { using Type = Vec3f; using TransformType = Vec3f; };
    template <> struct Vector<float, 4> { using Type = Vec4f; using TransformType = Vec3f; };

    template <typename T, uint32_t N> INLINE HOSTDEVICE bool          operator==(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
    template <typename T, uint32_t N> INLINE HOSTDEVICE bool          operator!=(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
    template <typename T, uint32_t N> INLINE HOSTDEVICE Matrix<T, N>  operator+(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
    template <typename T, uint32_t N> INLINE HOSTDEVICE Matrix<T, N>& operator+=(Matrix<T, N>& m1, const Matrix<T, N>& m2);
    template <typename T, uint32_t N> INLINE HOSTDEVICE Matrix<T, N>  operator-(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
    template <typename T, uint32_t N> INLINE HOSTDEVICE Matrix<T, N>& operator-=(Matrix<T, N>& m1, const Matrix<T, N>& m2);
    template <typename T, uint32_t N> INLINE HOSTDEVICE Matrix<T, N>  operator*(const Matrix<T, N>& m1, const Matrix<T, N>& m2);
    template <typename T, uint32_t N> INLINE HOSTDEVICE Matrix<T, N>  operator*(const Matrix<T, N>& m, const float t);
    template <typename T, uint32_t N> INLINE HOSTDEVICE Matrix<T, N>& operator*=(Matrix<T, N>& m1, const Matrix<T, N>& m2);
    template <typename T, uint32_t N> INLINE HOSTDEVICE Matrix<T, N>& operator*=(Matrix<T, N>& m, const float t);
    template <typename T, uint32_t N> INLINE HOSTDEVICE Matrix<T, N>  operator/(const Matrix<T, N>& m1, const float t);
    template <typename T, uint32_t N> INLINE HOSTDEVICE typename Matrix<T, N>::VecT operator*(const Matrix<T, N>& m, typename Matrix<T, N>::VecT& v);

    template <typename T> INLINE HOSTDEVICE Vec4f operator*(const Matrix<T, 3>& m, const Vec4f& v);
    template <typename T> INLINE HOSTDEVICE Vec3f operator*(const Matrix<T, 4>& m, const Vec3f& v);
 
    // Class definition
    template <typename T, uint32_t N>
    class Matrix
    {
    public:
        using VecT = typename Vector<T, N>::Type;
        using TransVecT = typename Vector<T, N>::TransformType;
        static constexpr uint32_t Dim = N;

        HOSTDEVICE Matrix();
        HOSTDEVICE Matrix(const Matrix& m);
        HOSTDEVICE Matrix(const T data[N*N]);
        HOSTDEVICE Matrix(const float (&data)[12]);
        HOSTDEVICE Matrix(const std::initializer_list<T>& list);

        HOSTDEVICE const T& operator[](uint32_t i) const;
        HOSTDEVICE       T& operator[](uint32_t i);

        HOSTDEVICE Matrix& operator=( const Matrix& a );

        HOSTDEVICE void setColumn(const VecT& v, uint32_t col_idx);
        HOSTDEVICE void setRow(const VecT& v, uint32_t row_idx);

        HOSTDEVICE const T& get(uint32_t i, uint32_t j) const;
        HOSTDEVICE       T& get(uint32_t i, uint32_t j);

        HOSTDEVICE void setData(const T data[N*N]);

        HOSTDEVICE       T* data();
        HOSTDEVICE const T* data() const;

        HOSTDEVICE bool isIdentity() const;

        HOSTDEVICE Matrix inverse() const;

        HOSTDEVICE Vec3f pointMul(const Vec3f& p) const;
        HOSTDEVICE Vec3f vectorMul(const Vec3f& v) const;
        HOSTDEVICE Vec3f normalMul(const Vec3f& n) const;

        static HOSTDEVICE Matrix           rotate(const float radians, const TransVecT& axis);
        static HOSTDEVICE Matrix<float, 4> translate(const Vec3f& t);
        static HOSTDEVICE Matrix<float, 4> translate(const float x, const float y, const float z);
        static HOSTDEVICE Matrix           scale(const TransVecT& s);
        static HOSTDEVICE Matrix           scale(const float s);
        static HOSTDEVICE Matrix<float, 4> shear(float a, float b, Axis axis);
        static HOSTDEVICE Matrix<float, 4> reflection(Axis axis);

        static HOSTDEVICE Matrix zero();
        static HOSTDEVICE Matrix identity();
    private:
        T m_data[N*N];
    };

    #ifndef __CUDACC__
    template <typename T, uint32_t N>
    inline std::ostream& operator<<(std::ostream& out, Matrix<T, N> m)
    {
        for (uint32_t i = 0; i < N*N; i++)
            out << m[i] << ' ';
        return out;
    }
    #endif // __CUDACC__

    // Operator overload
    // ----------------------------------------------------------------------------
    template <typename T, uint32_t N> 
    INLINE HOSTDEVICE bool operator==(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
    {
        bool is_equal = true;
        for (uint32_t i = 0; i < N*N; i++)
            is_equal &= (m1[i] == m2[i]);
        return is_equal;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE bool operator!=(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
    {
        return !(m1 == m2);
    }

    // ----------------------------------------------------------------------------
    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N> operator+(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
    {
        Matrix<T, N> ret(m1);
        ret += m2;
        return ret;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N>& operator+=(Matrix<T, N>& m1, const Matrix<T, N>& m2)
    {
        for (uint32_t i = 0; i < N*N; i++)
            m1[i] += m2[i];
        return m1;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N> operator-(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
    {
        Matrix<T, N> ret(m1);
        ret -= m2;
        return ret;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N>& operator-=(Matrix<T, N>& m1, const Matrix<T, N>& m2)
    {
        for (uint32_t i = 0; i < N*N; i++)
            m1[i] -= m2[i];
        return m1;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N> operator*(const Matrix<T, N>& m1, const Matrix<T, N>& m2)
    {
        Matrix<T, N> ret;
        for (uint32_t row = 0; row < N; row++)
        {
            for (uint32_t col = 0; col < N; col++)
            {
                T sum = static_cast<T>(0);
                for (uint32_t tmp = 0; tmp < N; tmp++)
                {
                    uint32_t m1_i = row * N + tmp;
                    uint32_t m2_i = tmp * N + col;
                    sum += m1[m1_i] * m2[m2_i];
                }
                ret[row * N + col] = sum;
            }
        }
        return ret;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N> operator*(const Matrix<T, N>& m, const float t)
    {
        Matrix<T, N> ret;
        for (uint32_t i = 0; i < N*N; i++)
            ret[i] = m[i] * t;
        return ret;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N>& operator*=(Matrix<T, N>& m1, const Matrix<T, N>& m2)
    {
        m1 = m1 * m2;
        return m1;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N>& operator*=(Matrix<T, N>& m1, const float t)
    {
        for (uint32_t i = 0; i < N*N; i++)
            m1[i] *= t;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N> operator/(const Matrix<T, N>& m1, const float t)
    {
        Matrix<T, N> ret;
        for (uint32_t i = 0; i < N*N; i++)
            ret[i] = m1[i] / t;
        return ret;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE typename Vector<T, N>::Type operator*(const Matrix<T, N>& m, const typename Matrix<T, N>::VecT& v)
    {
        typename Matrix<T, N>::VecT ret;

        for (uint32_t row = 0; row < N; row++)
        {
            T sum = static_cast<T>(0);
            for (uint32_t col = 0; col < N; col++)
            {
                sum += m[row * N + col] * v[col];
            }
            ret[row] = sum;
        }
        return ret;
    }

    template <typename T>
    INLINE HOSTDEVICE Vec4f operator*(const Matrix<T, 3>& m, const Vec4f& v)
    {
        Vec3f tmp(v);
        tmp = m * tmp;
        return Vec4f(tmp, 1.0f);
    }

    template <typename T>
    INLINE HOSTDEVICE Vec3f operator*(const Matrix<T, 4>& m, const Vec3f& v)
    {
        Vec4f tmp(v, 1.0f);
        tmp = m * tmp;
        return Vec3f(tmp);
    }


    // Class implementations
    // ----------------------------------------------------------------------------
    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N>::Matrix() : Matrix(Matrix::identity()) { }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N>::Matrix(const Matrix<T, N>& m)
    {
        for (uint32_t i = 0; i < N*N; i++)
            m_data[i] = m[i];
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N>::Matrix(const T data[N*N])
    {
        for (uint32_t i = 0; i < N * N; i++)
            m_data[i] = data[i];
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N>::Matrix(const float (&data)[12])
    {
        static_assert(std::is_same_v<T, float> && N == 4,
            "This constructor is only allowed for Matrix4f");

        for (uint32_t i = 0; i < 12; i++)
            m_data[i] = data[i];
        m_data[12] = m_data[13] = m_data[14] = 0.0f;
        m_data[15] = 1.0f;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N>::Matrix(const std::initializer_list<T>& list)
    {
        uint32_t i = 0; 
        for (auto it = list.begin(); it != list.end(); ++it)
            m_data[ i++ ] = *it;
    }

    // ----------------------------------------------------------------------------
    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N>& Matrix<T, N>::operator=(const Matrix<T, N>& a)
    {
        for (uint32_t i = 0; i < N * N; i++)
            m_data[i] = a[i];
        return *this;
    }

    // ----------------------------------------------------------------------------
    template <typename T, uint32_t N>
    INLINE HOSTDEVICE const T& Matrix<T, N>::operator[](const uint32_t i) const
    {
        return m_data[i];
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE T& Matrix<T, N>::operator[](const uint32_t i)
    {
        return m_data[i];
    }

    // ----------------------------------------------------------------------------
    template <typename T, uint32_t N>
    INLINE HOSTDEVICE const T& Matrix<T, N>::get(uint32_t i, uint32_t j) const
    {
        uint32_t idx = i * N + j;
        return m_data[idx];
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE T& Matrix<T, N>::get(uint32_t i, uint32_t j)
    {
        uint32_t idx = i * N + j;
        return m_data[idx];
    }

    // ----------------------------------------------------------------------------
    template <typename T, uint32_t N>
    INLINE HOSTDEVICE void Matrix<T, N>::setData(const T data[N*N])
    {
        for (uint32_t i = 0; i < N*N; i++)
            m_data[i] = data[i];
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE T* Matrix<T, N>::data()
    {
        return m_data;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE const T* Matrix<T, N>::data() const
    {
        return m_data;
    }

    // ----------------------------------------------------------------------------
    template <typename T, uint32_t N>
    INLINE HOSTDEVICE bool Matrix<T, N>::isIdentity() const
    {
        return *this == Matrix<T, N>::identity();
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N> Matrix<T, N>::inverse() const
    {
        Matrix<T, N> ret = Matrix<T, N>::identity();
        Matrix<T, N> mat(*this);
        float tmp;
        for (uint32_t i = 0; i < N; i++)
        {
            tmp = 1.0f / mat[i * N + i];
            for (uint32_t j = 0; j < N; j++)
            {
                mat[i * N + j] *= tmp;
                ret[i * N + j] *= tmp;
            }
            for (uint32_t j = 0; j < N; j++)
            {
                if (i != j)
                {
                    tmp = mat[j * N + i];
                    for (uint32_t k = 0; k < N; k++)
                    {
                        mat[j * N + k] -= mat[i * N + k] * tmp;
                        ret[j * N + k] -= ret[i * N + k] * tmp;
                    }
                }
            }
        }
        return ret;
    }

    // ----------------------------------------------------------------------------
    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Vec3f Matrix<T, N>::pointMul(const Vec3f& p) const 
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<N, 4>);
        return Vec3f(0.0f);
    }

    template<>
    INLINE HOSTDEVICE Vec3f Matrix<float, 4>::pointMul(const Vec3f& p) const
    {
        float x = m_data[0]  * p.x() + m_data[1]  * p.y() + m_data[2]  * p.z() + m_data[3];
        float y = m_data[4]  * p.x() + m_data[5]  * p.y() + m_data[6]  * p.z() + m_data[7];
        float z = m_data[8]  * p.x() + m_data[9]  * p.y() + m_data[10] * p.z() + m_data[11];
        float w = m_data[12] * p.x() + m_data[13] * p.y() + m_data[14] * p.z() + m_data[15];

        if (w == 1.0f || w == 0.0f)
            return Vec3f(x, y, z);
        else
            return Vec3f(x, y, z) / w;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Vec3f Matrix<T, N>::vectorMul(const Vec3f& n) const 
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<N, 4>);
        return Vec3f(0.0f);
    }

    template<>
    INLINE HOSTDEVICE Vec3f Matrix<float, 4>::vectorMul(const Vec3f& v) const
    {
        return Vec3f(
            m_data[0] * v.x() + m_data[1] * v.y() + m_data[2]  * v.z(), 
            m_data[4] * v.x() + m_data[5] * v.y() + m_data[6]  * v.z(),
            m_data[8] * v.x() + m_data[9] * v.y() + m_data[10] * v.z()
        );
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Vec3f Matrix<T, N>::normalMul(const Vec3f& n) const 
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<N, 4>);
        return Vec3f(0.0f);
    }

    template<>
    INLINE HOSTDEVICE Vec3f Matrix<float, 4>::normalMul(const Vec3f& n) const
    {   
        Matrix4f inv = this->inverse();

        float x = n.x(), y = n.y(), z = n.z();
        return Vec3f(
            inv[0] * x + inv[4] * y + inv[8]  * z,
            inv[1] * x + inv[5] * y + inv[9]  * z,
            inv[2] * x + inv[6] * y + inv[10] * z
        );
    }

    // ----------------------------------------------------------------------------
    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N> Matrix<T, N>::rotate(const float radians, const typename Matrix<T, N>::TransVecT& axis)
    {
        static_assert(1 < N && N <= 4, "Matrix::rotate(): The dimension of matrix must be 2x2, 3x3 or 4x4.");

        const float s = sinf(radians);
        const float c = cosf(radians);

        Matrix<T, N> i_mat = Matrix<T, N>::identity();

        if constexpr (N == 2)
        {
            i_mat[0] = c; i_mat[1] = -s; 
            i_mat[2] = s; i_mat[3] = c;
        }
        else if constexpr (N == 3 || N == 4)
        {
            Matrix<T, N>::TransVecT a = normalize(axis);
            // 1st row
            i_mat[0] = a.x() * a.x() + (1.0 - a.x() * a.x()) * c;
            i_mat[1] = a.x() * a.y() * (1.0 - c) - a.z() * s;
            i_mat[2] = a.x() * a.z() * (1.0 - c) + a.y() * s;

            // 2nd row
            i_mat[1 * N + 0] = a.x() * a.y() * (1.0 - c) + a.z() * s;
            i_mat[1 * N + 1] = a.y() * a.y() + (1.0 - a.y() * a.y()) * c;
            i_mat[1 * N + 2] = a.y() * a.z() * (1.0 - c) - a.x() * s;

            // 3rd row
            i_mat[2 * N + 0] = a.x() * a.z() * (1.0 - c) - a.y() * s;  
            i_mat[2 * N + 1] = a.y() * a.z() * (1.0 - c) + a.x() * s;
            i_mat[2 * N + 2] = a.z() * a.z() + (1.0 - a.z() * a.z()) * c;
        }
        return i_mat;
    }

    template <>
    INLINE HOSTDEVICE Matrix<float, 4> Matrix<float, 4>::translate(const Vec3f &t)
    {
        Matrix4f i_mat = Matrix4f::identity();
        i_mat[0 * 4 + 3] = t.x();
        i_mat[1 * 4 + 3] = t.y();
        i_mat[2 * 4 + 3] = t.z();
        return i_mat;
    }

    template <>
    INLINE HOSTDEVICE Matrix<float, 4> Matrix<float, 4>::translate(const float x, const float y, const float z)
    {
        return Matrix4f::translate(Vec3f(x, y, z));
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N> Matrix<T, N>::scale(const typename Matrix<T, N>::TransVecT& s)
    {
        static_assert(1 < N && N <= 4, "Matrix::scale(): The dimension of matrix must be 2x2, 3x3 or 4x4.");

        Matrix<T, N> i_mat = Matrix<T, N>::identity();
        for (size_t i = 0; i < static_cast<size_t>(sizeof(s) / sizeof(float)); i++)
            i_mat[i * N + i] = s[i];
        return i_mat;
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N> Matrix<T, N>::scale(const float s)
    {
        static_assert(1 < N && N <= 4, "Matrix::scale(): The dimension of matrix must be 2x2, 3x3 or 4x4.");

        Matrix<T, N> i_mat = Matrix<T, N>::identity();
        for (size_t i = 0; i < static_cast<size_t>(sizeof(Matrix<T, N>::TransVecT) / sizeof(float)); i++)
            i_mat[i * N + i] = s;
        return i_mat;
    }

    template <>
    INLINE HOSTDEVICE Matrix<float, 4> Matrix<float, 4>::reflection(Axis axis)
    {
        Matrix4f i_mat = Matrix4f::identity();
        uint32_t i = static_cast<uint32_t>(axis);
        i_mat[i * 4 + 1] *= -1;
        return i_mat;
    }

    template <>
    INLINE HOSTDEVICE Matrix<float, 4> Matrix<float, 4>::shear(float a, float b, Axis axis)
    {
        Matrix4f i_mat = Matrix4f::identity();
        switch(axis)
        {
            case Axis::X:
                i_mat[4] = a; i_mat[8] = b;
            case Axis::Y:
                i_mat[1] = a; i_mat[9] = b;
            case Axis::Z:
                i_mat[2] = a; i_mat[6] = b;      
        }
        return i_mat;
    }

    // ----------------------------------------------------------------------------
    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N> Matrix<T, N>::zero()
    {
        T data[N*N];
        for (uint32_t i = 0; i < N*N; i++)
            data[i] = 0;
        return Matrix<T, N>(data);
    }

    template <typename T, uint32_t N>
    INLINE HOSTDEVICE Matrix<T, N> Matrix<T, N>::identity()
    {
        T data[N*N];
        for (uint32_t i = 0; i < N; i++)
        {
            for (uint32_t j = 0; j < N; j++)
            {
                data[i * N + j] = i == j ? static_cast<T>(1) : static_cast<T>(0);
            }
        }
        return Matrix<T, N>(data);
    }

} // ::prayground