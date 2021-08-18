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
#include <variant>
#include "../core/stream_helpers.h"

#if defined(_WIN32) | defined(_WIN64)
    #include <windows.h>
#endif

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

enum MessageType
{
    MSG_NORMAL,
    MSG_WARNING,
    MSG_ERROR
};

template <class T>
concept BoolConvertable = requires(T x)
{
    (bool)x;
};

inline void Throw(const std::string& msg) {
    throw std::runtime_error(msg);
}

template <BoolConvertable T>
inline void Assert(T condition, const std::string& msg) {
    if (!(bool)condition) Throw(msg);
}

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

#endif  // defined(__linux__)

    // Recusrive call of message function  
    const size_t num_args = sizeof...(args);
    if constexpr (num_args > 0) {
        std::cout << ' ';
        Message(type, args...);
    }
    if constexpr (num_args == 0) std::cout << std::endl;
}

/** 実装してない関数が多すぎるので、マクロ設定で簡略化する。横着です。 */
#define TODO_MESSAGE()                                  \
    std::stringstream ss;                               \
    ss << "Sorry! The function you called at "          \
       << "' (" __FILE__ << ":" << __LINE__ << ")"      \
       << " is still under development! ";              \
    Message(MSG_WARNING, ss.str());

#endif // __CUDACC__

}
