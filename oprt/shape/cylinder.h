#pragma once 

#include "optix/cylinder.cuh"

#ifndef __CUDACC__
#include "../core/shape.h"
#include "../core/cudabuffer.h"

namespace oprt {

class Cylinder final : public Shape {
public:
    explicit Cylinder(float radius, float height) {}

    ShapeType type() const override { return ShapeType::Cylinder; }

private:
    float m_radius;
    float m_height;
};

}

#endif