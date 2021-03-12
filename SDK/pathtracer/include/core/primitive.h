#pragma once

#include <core/util.h>
#include <core/shape.h>
#include <core/material.h>
#include <core/transform.h>

namespace pt {

class Primitive {
private:
    ShapePtr shape_ptr;
    MaterialPtr material_ptr;
    Transform* transform;

public:
    Primitive(ShapePtr shape, MaterialPtr material, Transform* transform)
    : shape_ptr(shape), material_ptr(material), transform(transform) {}

    ShapeType get_shapetype() const { return shape_ptr->type(); }
    MaterialType get_materialtype() const { return material_ptr->type(); }
};

}