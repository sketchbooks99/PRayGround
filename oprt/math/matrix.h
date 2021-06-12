#include "vec.h"

template <size_t M, size_t N>
class Matrix
{
public:
    float operator[](int idx) {}
private:
    float m[M*N];
};

using Matrix4f = Matrix<4,4>;
using Matrix3f = Matrix<3,3>;
using Matrix3x4 = Matrix<3,4>;