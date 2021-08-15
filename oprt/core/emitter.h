#pragma once

#include "util.h"

namespace oprt {

enum class EmitterType {
    Area = 0, 
    Point = 1,
    Envmap = 2,
    Count = 3
};

#ifndef __CUDACC__

inline std::ostream& operator<<(std::ostream& out, EmitterType type)
{
    switch(type)
    {
        case EmitterType::Point:   return out << "EmitterType::Point";
        case EmitterType::Area:    return out << "EmitterType::Area";
        case EmitterType::Envmap:  return out << "EmitterType::Envmap";
        default:                   return out << "";
    }
}

class Emitter {
public:
    virtual void prepareData() = 0;
    virtual EmitterType type() const = 0;

    virtual void freeData() = 0;

    void* devicePtr() const { return reinterpret_cast<void*>(d_data); }
protected:
    void* d_data { nullptr };
};

#endif // __CUDACC__

} // ::oprt
