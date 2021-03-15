#pragma once

#include <core/util.h>
#include <core/shape.h>
#include <core/material.h>
#include <core/transform.h>
#include <optix/program.h>

namespace pt {

class Primitive {
public:
    Primitive(ShapePtr shape_ptr, MaterialPtr material_ptr, const Transform& transform, uint32_t sbt_index)
    : m_shape_ptr(shape_ptr), m_material_ptr(material_ptr), m_transform(m_transform), sbt_index(sbt_index) {}
    ShapeType get_shapetype() const { return m_shape_ptr->type(); }
    MaterialType get_materialtype() const { return m_material_ptr->type(); }
private:
    ShapePtr m_shape_ptr;
    MaterialPtr m_material_ptr;
    Transform m_transform;
    ProgramGroup m_program_group;
    uint32_t sbt_index { 0 }; // For managing sbt index which associated with a shader.
};

}