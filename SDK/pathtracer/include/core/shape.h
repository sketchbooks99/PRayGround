#pragma once

#include <core/util.h>
#include <sutil/vec_math.h>

namespace pt {

enum class ShapeType {
    None,       // None type
    Mesh,       // Mesh with triangle 
    Sphere,     // Sphere 
    Plane       // Plane (rectangle)
};

#ifndef __CUDACC__

inline std::ostream& operator<<(std::ostream& out, ShapeType type) {
    switch(type) {
    case ShapeType::None:
        return out << "ShapeType::None";
    case ShapeType::Mesh:
        return out << "ShapeType::Mesh";
    case ShapeType::Sphere:
        return out << "ShapeType::Sphere";
    case ShapeType::Plane:
        return out << "ShapeType::Plane";
    default:
        return out << "";
    }
}
#endif

// Abstract class for readability
class Shape {
public:
    virtual HOST ShapeType type() const = 0;
    virtual HOST void build_input( OptixBuildInput& bi, uint32_t sbt_idx ) const = 0;
};

using ShapePtr = Shape*;

}