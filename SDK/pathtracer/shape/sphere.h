#pragma once 

#include "../core/shape.h"
#include "../core/transform.h"

namespace pt {

struct SphereData {
    float radius;
    sutil::Transform transform;
};

#if !defined(__CUDACC__)
class Sphere : public Shape {
public:
    explicit Sphere(float r) : radius(r) {}

    ShapeType type() const override { return ShapeType::Sphere; }
private:
    float radius;
};
#endif

}