#pragma once

#if !defined(__CUDACC__)
#include <sutil/vec_math.h>
#include "core_util.h"
#endif

namespace pt {

enum class ShapeType {
    None,       // None type
    Mesh,       // Mesh with triangle 
    Sphere,     // Sphere 
    Plane       // Plane (rectangle)
};

#if !defined(__CUDACC__)
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
        Throw("This ShapeType is not supported\n");
        return out << "";
    }
}
#endif

// Abstract class for readability
struct Shape {
    virtual ShapeType type() const = 0;
};

// Mesh Data
struct Mesh : public Shape {
    float4* vertices;
    float4* normals;
    int3* indices;

    ShapeType type() const override { return ShapeType::Mesh; }
};

struct Sphere : public Shape {
    float radius;

    ShapeType type() const override { return ShapeType::Sphere; }
};

struct Plane : public Shape {
    float size;
    
    ShapeType type() const override { return ShapeType::Plane; }
};

}