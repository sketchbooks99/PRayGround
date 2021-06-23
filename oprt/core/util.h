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
#include <filesystem>
#include <optional>
#include <map>
#include <concepts>
#include "../core/stream_helpers.h"

#if defined(_WIN32) | defined(_WIN64)
    #include <windows.h>
#endif

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

enum MessageType
{
    MSG_NORMAL,
    MSG_WARNING,
    MSG_ERROR
};

inline void Throw(const std::string& msg) {
    throw std::runtime_error(msg);
}

template <class T>
concept BoolConvertible = requires(T x)
{
    (bool)x;
};

template <BoolConvertible T>
inline void Assert(T condition, const std::string& msg) {
    if (!(bool)condition) Throw(msg);
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

template <typename Head, typename... Args>
inline void Message(MessageType type, Head head, Args... args) {

#if defined(__linux__)

    switch(type)
    {
        case MSG_NORMAL:
            break;
        case MSG_WARNING:
            std::cout << "\033[33m"; // yellow
            break;
        case MSG_ERROR:
            std::cout << "\033[31m"; // red
            break;
    }
    std::cout << head << "\033[0m";

#elif defined(_WIN32) | defined(_WIN64)

    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO consoleInfo;
    WORD current_attributes;

    GetConsoleScreenBufferInfo(hConsole, &consoleInfo);
    current_attributes = consoleInfo.wAttributes;
    switch (type)
    {
        case MSG_NORMAL:
            break;
        case MSG_WARNING:
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN); // yellow
            break;
        case MSG_ERROR:
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED);                    // red
            break;
    }
    std::cout << head;
    SetConsoleTextAttribute(hConsole, current_attributes);

#endif

    // Recusrive call of message function  
    const size_t num_args = sizeof...(args);
    if constexpr (num_args > 0) {
        std::cout << ' ';
        Message(type, args...);
    }
    if constexpr (num_args == 0) std::cout << std::endl;
}

#endif

}
