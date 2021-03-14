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
    Transform transform;

    uint32_t sbt_index { 0 }; // For managing sbt index which associated with a shader.

public:
    Primitive(ShapePtr shape, MaterialPtr material, const Transform& transform, uint32_t sbt_index)
    : shape_ptr(shape), material_ptr(material), transform(transform), sbt_index(sbt_index) {}

    ShapeType get_shapetype() const { return shape_ptr->type(); }
    MaterialType get_materialtype() const { return material_ptr->type(); }
};

}