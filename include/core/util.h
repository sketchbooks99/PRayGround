#pragma once 

#ifndef __CUDACC__
#include <sutil/Exception.h>
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>
#include <vector>
#include <include/core/stream_helpers.h>
#endif

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
 * \note
 * Which code is better for readability?
 * 1. Use recursive call as follows.
 * 2. Use recursive call but doesn't prepare the dummy function as like Message(). 
 *    Instead, switch conditions due to the number of args as like follows.
 *    if constexpr (sizeof...(args) > 0) Message(args...);
 *    if constexpr (sizeof...(args) == 0) std::endl;
 */

template <typename T>
inline void Message_once(T t) {
    std::cout << t;
}

template <typename Head, typename... Args>
inline void Message(Head head, Args... args) {
    Message_once(head);
    const size_t num_args = sizeof...(args);
    if constexpr (num_args > 0) {
        std::cout << ' ';
        Message(args...);
    }
    if constexpr (num_args == 0) std::cout << std::endl;
}

/**
 * \brief 
 * Dummy free function for end of recursive cuda_free() using variadic templates.
 */ 
inline void cuda_frees() {}

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

}
