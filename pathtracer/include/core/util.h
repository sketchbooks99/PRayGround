#pragma once 

#include <sutil/Exception.h>
#include <string>
#include <stdexcept>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>
#include <vector>
#include <core/stream_helpers.h>

namespace pt {

template <typename T>
void swap(T& a, T& b)
{
#ifdef __CUDACC__
    T c(a); a = b; b = c;
#else   
    std::swap(a, b);
#endif
}

/** Error handling at the host side. */
#ifndef __CUDACC__
inline void Throw(const std::string& msg) {
    throw std::runtime_error(msg);
}

inline void Assert(bool condition, const std::string& msg) {
    if (!condition) Throw(msg);
}

/**
 * \brief 
 * Dummy free function for end of recursive cuda_free() using variadic templates.
 */ 
inline void cuda_free() {}

template <typename T>
inline void cuda_free(T data) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(data)));
}

/** 
 * \brief 
 * Recursive free of object.
 */
template <typename Head, typename... Args>
inline void cuda_frees(Head head, Args... args) {
    cuda_free(head);
    cuda_frees(args...);
}
#endif

/** 
 * \brief Initialize object on device.
 * 
 * \note Initailization must be excecuted only once.
 */
template <typename T, typename... Args>
__global__ void setup_object_on_device(T** d_ptr, Args... args) {
    (*d_ptr) = new T(args...);
}

template <typename T>
__global__ void delete_object_on_device(T** d_ptr) {
    delete (void*)*d_ptr;
}

}
