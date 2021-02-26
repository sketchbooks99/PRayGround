#pragma once 

#include <string>
#include <stdexcept>

namespace pt {

inline void Throw(const std::string& msg) {
    throw std::runtime_error(msg);
}

inline void Assert(bool condition, const std::string& msg) {
    if(!condition) Throw(msg);
}

}