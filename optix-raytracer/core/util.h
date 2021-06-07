#pragma once 

#ifndef __CUDACC__
#include <sutil/Exception.h>
#include <string>
#include <cuda_runtime.h>
#include <stdexcept>
#include <array>
#include <regex>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>
#include <vector>
#include <utility>
#include "../core/stream_helpers.h"
#endif

#include "../optix/macros.h"

namespace oprt {

struct CameraData {
    float3 eye;
    float3 U;
    float3 V;
    float3 W;
    float aperture;
};

// Parameters are configured when ray tracing on device is launched. 
struct Params
{
    unsigned int subframe_index;
    float4* accum_buffer;
    uchar4* frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    unsigned int max_depth;

    // CameraData camera;
    float3 eye;
    float3 U;
    float3 V; 
    float3 W; 
    float aperture;

    OptixTraversableHandle handle; // unsigned long long
};

enum HitType
{
    HIT_OUTSIDE_FROM_OUTSIDE = 1u << 0,
    HIT_OUTSIDE_FROM_INSIDE = 1u << 1,
    HIT_INSIDE_FROM_OUTSIDE = 1u << 2,
    HIT_INSIDE_FROM_INSIDE = 1u << 3
};

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
inline void MessageOnce(T t) {
    std::cout << t;
}
template <typename Head, typename... Args>
inline void Message(Head head, Args... args) {
    MessageOnce(head);
    const size_t num_args = sizeof...(args);
    if constexpr (num_args > 0) {
        std::cout << ' ';
        Message(args...);
    }
    if constexpr (num_args == 0) std::cout << std::endl;
}

#endif

}
