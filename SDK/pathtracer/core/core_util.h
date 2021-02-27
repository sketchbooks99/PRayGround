#pragma once 

#if !defined(__CUDACC__)
#include <string>
#include <stdexcept>

inline void Throw(const std::string& msg) {
    throw std::runtime_error(msg);
}

inline void Assert(bool condition, const std::string& msg) {
    if(!condition) Throw(msg);
}

#endif
