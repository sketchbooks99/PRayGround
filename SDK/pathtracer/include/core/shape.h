#pragma once

#include <core/util.h>
#include <core/aabb.h>
#include <sutil/vec_math.h>

namespace pt {

enum class ShapeType {
    None,       // None type
    Mesh,       // Mesh with triangle 
    Sphere,     // Sphere 
    Plane       // Plane (rectangle)
};

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

// Abstract class for readability
class Shape {
public:
    virtual HOST ShapeType type() const = 0;
    virtual HOST void prepare_shapedata() const = 0;
    virtual HOST void build_input( OptixBuildInput& bi, uint32_t sbt_idx ) const = 0;
    virtual HOST AABB bound() const = 0;

protected:
    CUdeviceptr d_data_ptr;
};

using ShapePtr = Shape*;

}