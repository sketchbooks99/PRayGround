#pragma once

#include "util.h"
#include "shape.h"
#include "texture.h"

namespace oprt {

enum class EmitterType {
    Point = 0,
    Area = 1, 
    Envmap = 2,
    Count = 3
};

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

    void* devicePtr() const { return d_data; }
protected:
    void* d_data { 0 };
};

}
