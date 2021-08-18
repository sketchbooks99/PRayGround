#pragma once

#include <vector_functions.h>
#include <vector_types.h>

#ifndef __CUDACC__
#include <cmath>
#include <cstdlib>
#endif

template <typename T, uint32_t N> class Vector;
using Vector2f = Vector<float, 2>;
using Vector3f = Vector<float, 3>;
using Vector4f = Vector<float, 4>;

template <typename T, uint32_t N> 

template <typename T, uint32_t N>
class Vector 
{
public:

private:
    T[N] m_data;
};