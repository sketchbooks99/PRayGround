#pragma once 

#ifndef __CUDACC__
#include <sutil/Exception.h>
#include <string>
#include <cuda_runtime.h>
#include <stdexcept>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>
#include <vector>
#include "../core/stream_helpers.h"
#endif

#include "../optix/macros.h"

namespace oprt {

enum class Axis {
    X = 0, 
    Y = 1, 
    Z = 2
};

/** Error handling at the host side. */
#ifndef __CUDACC__
inline void Throw(const std::string& msg) {
    throw std::runtime_error(msg);
}

inline void Assert(bool condition, const std::string& msg) {
    if (!condition) Throw(msg);
}

template <typename T>
inline void cuda_free(T& data) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(data)));
}
/** 
 * \brief 
 * Recursive free of object.
 */
template <typename Head, typename... Args>
inline void cuda_frees(Head& head, Args... args) {
    cuda_free(head);
    if constexpr (sizeof...(args) > 0) 
        cuda_frees(args...);
}

/**
 * @brief Stream out object recursively. 
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

#endif

}
