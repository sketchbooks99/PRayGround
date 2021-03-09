#pragma once 

#include "../core/shape.h"

class Sphere : public Shape {
public:
    Sphere(float r) : radius(r) {}

    ShapeType type() const override { return ShapeType::Sphere; }
private:
    float radius;
};