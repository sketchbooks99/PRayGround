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

/** Error handling at the host side. */
#ifndef __CUDACC__

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

#define UNIMPLEMENTED()                                                     \
    do {                                                                    \
        std::stringstream ss;                                               \
        ss << "The function you called at "                                 \
           << " (" __FILE__ << ":" << __LINE__ << ")"                       \
           << " will not be implemented or is still under development";     \
        Message(MSG_WARNING, ss.str());                                     \
    } while (0)

#define LOG(msg, ...)                                                    \
    do {                                                                 \
        std::stringstream ss;                                            \
        ss << msg;                                                       \
        Message(MSG_NORMAL, ss.str() __VA_OPT__(,) __VA_ARGS__);         \
    } while (0)

#define LOG_WARN(msg, ...)                                                          \
    do {                                                                            \
    std::stringstream ss;                                                           \
        ss << "[Warning] " << msg;                                                  \
        Message(MSG_WARNING, ss.str() __VA_OPT__(,) __VA_ARGS__);                   \
    } while (0)

#define LOG_FATAL(msg, ...)                                                     \
    do {                                                                        \
        std::stringstream ss;                                                   \
        ss << "[Fatal] " << msg;                                                \
        Message(MSG_FATAL, ss.str() __VA_OPT__(,) __VA_ARGS__);                 \
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
               << "' (" __FILE__ << ":" << __LINE__ << ")"  \
               << ", " << msg;                              \
            throw std::runtime_error(ss.str());             \
        }                                                   \
    } while (0)

#endif // __CUDACC__

}
