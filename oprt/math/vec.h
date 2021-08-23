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