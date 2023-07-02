#pragma once 

#ifndef __CUDACC__
#include <string>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <sstream>
#include <concepts>
#include <vector>
#include <array>
#include <iostream>
#include <memory>
#include <assert.h>
#include <prayground/core/stream_helpers.h>

#if defined(_WIN32) | defined(_WIN64)
#include <windows.h>
#endif // _WIN32 | _WIN64

#endif // __CUDACC__

#include <prayground/optix/macros.h>

namespace prayground {

/* Declare integer types for CUDA kernel */
#ifdef __CUDACC__
#if !defined(int8_t)
    typedef signed char int8_t;
#endif

#if !defined(int16_t)
    typedef short int16_t;
#endif

#if !defined(int32_t)
    typedef int int32_t;
#endif

#if !defined(int64_t)
    typedef long long int64_t;
#endif

#if !defined(uint8_t)
    typedef unsigned char uint8_t;
#endif

#if !defined(uint16_t)
    typedef unsigned short uint16_t;
#endif

#if !defined(uint32_t)
    typedef unsigned int uint32_t;
#endif 

#if !defined(uint64_t)
    typedef unsigned long long uint64_t;
#endif

#else // __CUDACC__

enum MessageType
{
    MSG_NORMAL,
    MSG_WARNING,
    MSG_FATAL
};

template <typename T>
inline void cuda_free(T& data) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(data)));
}
/** @brief Recursive release of object from a device. */
template <typename Head, typename... Args>
inline void cuda_frees(Head& head, Args... args) {
    cuda_free(head);
    if constexpr (sizeof...(args) > 0) 
        cuda_frees(args...);
}

/** @brief Recursively print object to a standard stream. */
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
        case MSG_FATAL:
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
        case MSG_FATAL:
            SetConsoleTextAttribute(hConsole, FOREGROUND_RED);                    // red
            break;
    }
    std::cout << head;
    SetConsoleTextAttribute(hConsole, current_attributes);

#endif  // defined(__linux__)

    // Recusrive call of message function  
    const size_t num_args = sizeof...(args);
    if constexpr (num_args > 0) {
        std::cout << ' ';
        Message(type, args...);
    }
    if constexpr (num_args == 0) std::cout << std::endl;
}

template <typename Head, typename... Args>
inline void pgLog(Head head, Args... args) { Message(MSG_NORMAL, head, args...); }

template <typename Head, typename... Args>
inline void pgLogWarn(Head head, Args... args) { Message(MSG_WARNING, "[Warning]", head, args...); }

template <typename Head, typename... Args>
inline void pgLogFatal(Head head, Args... args) { Message(MSG_FATAL, "[Fatal]", head, args...); }

#define UNIMPLEMENTED()                                                     \
    do {                                                                    \
        std::stringstream ss;                                               \
        ss << "The function you called at "                                 \
           << " (" __FILE__ << ":" << __LINE__ << ")"                       \
           << " will not be implemented or is still under development";     \
        Message(MSG_WARNING, ss.str());                                     \
    } while (0)

/** アサーション用 */
#define THROW(msg)                                          \
    do {                                                    \
        std::stringstream ss;                               \
        ss << "'(" __FILE__ << ":" << __LINE__ << ")"       \
           << ", " << msg;                                  \
        throw std::runtime_error(ss.str());                 \
    } while (0)

#define ASSERT(cond, msg)                                   \
    do {                                                    \
        if (!(bool)(cond)) {                                \
            std::stringstream ss;                           \
            ss << "Assertion failed at "                    \
               << "(" __FILE__ << ":" << __LINE__ << ")"    \
               << ", " << msg;                              \
            throw std::runtime_error(ss.str());             \
        }                                                   \
    } while (0)

#define PG_LOG(msg, ...)                                        \
    do {                                                        \
        std::stringstream ss;                                   \
        ss << "(" __FILE__ << ":" << __LINE__ << ") :";         \
        pgLog(ss.str(), msg, ##__VA_ARGS__);        \
    }                                                           \
    while (0)

#define PG_LOG_WARN(msg, ...)                                   \
    do {                                                        \
        std::stringstream ss;                                   \
        ss << "(" __FILE__ << ":" << __LINE__ << ") :";         \
        pgLogWarn(ss.str(), msg, ##__VA_ARGS__);       \
    }                                                           \
    while (0)

#define PG_LOG_FATAL(msg, ...)                                  \
    do {                                                        \
        std::stringstream ss;                                   \
        ss << "(" __FILE__ << ":" << __LINE__ << ") :";         \
        pgLogFatal(ss.str(), msg, ##__VA_ARGS__);         \
    }                                                           \
    while (0)

#endif // __CUDACC__

} // namespace prayground
