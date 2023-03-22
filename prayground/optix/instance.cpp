#include "instance.h"
#include <prayground/core/cudabuffer.h>

namespace prayground {

// ------------------------------------------------------------------
Instance::Instance()
{
    m_instance = new OptixInstance;
    setTransform(Matrix4f::identity());
    m_instance->flags = OPTIX_INSTANCE_FLAG_NONE;
    m_instance->visibilityMask = 255;
}

Instance::Instance(const Matrix4f& matrix)
{
    m_instance = new OptixInstance;
    setTransform(matrix);
    m_instance->flags = OPTIX_INSTANCE_FLAG_NONE;
    m_instance->visibilityMask = 255;
}

Instance::Instance(const Instance& instance)
{
    m_instance = instance.rawInstancePtr();
}

// ------------------------------------------------------------------
void Instance::setId(const uint32_t id)
{
    m_instance->instanceId = id;
}

void Instance::setSBTOffset(const uint32_t sbt_offset)
{
    m_instance->sbtOffset = sbt_offset;
}

void Instance::setVisibilityMask(const uint32_t visibility_mask)
{
    m_instance->visibilityMask = visibility_mask;
}

void Instance::setTraversableHandle(OptixTraversableHandle handle)
{
    m_instance->traversableHandle = handle;
}

void Instance::setPadding(uint32_t pad[2])
{
    memcpy(m_instance->pad, pad, sizeof(uint32_t) * 2);
}

void Instance::setFlags(const uint32_t flags)
{
    m_instance->flags = flags;
}

// ------------------------------------------------------------------
uint32_t Instance::id() const
{
    return m_instance->instanceId;
}

uint32_t Instance::sbtOffset() const
{
    return m_instance->sbtOffset;
}

uint32_t Instance::visibilityMask() const
{
    return m_instance->visibilityMask;
}
OptixTraversableHandle Instance::handle() const
{
    return m_instance->traversableHandle;
}

uint32_t Instance::flags() const
{
    return m_instance->flags;
}

// ------------------------------------------------------------------
void Instance::setTransform(const Matrix4f& matrix)
{
    memcpy(m_instance->transform, matrix.data(), sizeof(float)*12);
}

void Instance::translate(const Vec3f& t)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::translate(t);
    setTransform(current_mat);
}

void Instance::scale(const Vec3f& s)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::scale(s);
    setTransform(current_mat);
}

void Instance::scale(const float s)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::scale(s);
    setTransform(current_mat);
}

void Instance::rotate(const float radians, const Vec3f& axis)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::rotate(radians, axis);
    setTransform(current_mat);
}

void Instance::rotateX(const float radians)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::rotate(radians, Vec3f(1.0f, 0.0f, 0.0f));
    setTransform(current_mat);
}

void Instance::rotateY(const float radians)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::rotate(radians, Vec3f(0.0f, 1.0f, 0.0f));
    setTransform(current_mat);
}

void Instance::rotateZ(const float radians)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::rotate(radians, Vec3f(0.0f, 0.0f, 1.0f));
    setTransform(current_mat);
}

void Instance::reflect(Axis axis)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::reflection(axis);
    setTransform(current_mat);
}

Matrix4f Instance::transform()
{
    return Matrix4f(m_instance->transform);
}

// ------------------------------------------------------------------
OptixInstance* Instance::rawInstancePtr() const
{
    return m_instance;
}

/********************************************************************
 ShapeInstance                                                     
********************************************************************/
// ------------------------------------------------------------------
ShapeInstance::ShapeInstance(ShapeType type)
: m_type{type}, m_instance(Instance{}), m_gas{type}
{

}

ShapeInstance::ShapeInstance(ShapeType type, const Matrix4f& m)
: m_type{ type }, m_instance(Instance(m)), m_gas{type}
{

}

ShapeInstance::ShapeInstance(ShapeType type, const std::shared_ptr<Shape>& shape)
: m_type{ type }, m_instance(Instance{}), m_gas{ type }
{
    m_gas.addShape(shape);
}

ShapeInstance::ShapeInstance(ShapeType type, const std::shared_ptr<Shape>& shape, const Matrix4f& m)
: m_type{ type }, m_instance(Instance(m)), m_gas{ type }
{
    m_gas.addShape(shape);
}

// ------------------------------------------------------------------
void ShapeInstance::addShape(const std::shared_ptr<Shape>& shape)
{
    m_gas.addShape(shape);
}

std::vector<std::shared_ptr<Shape>> ShapeInstance::shapes() const 
{
    return m_gas.shapes();
}

// ------------------------------------------------------------------
void ShapeInstance::setId(const uint32_t id)
{
    m_instance.setId(id);
}

void ShapeInstance::setSBTOffset(const uint32_t sbt_offset)
{
    m_instance.setSBTOffset(sbt_offset);
}

void ShapeInstance::setVisibilityMask(const uint32_t visibility_mask)
{
    m_instance.setVisibilityMask(visibility_mask);
}

void ShapeInstance::setPadding(uint32_t pad[2])
{
    m_instance.setPadding(pad);
}

void ShapeInstance::setFlags(const uint32_t flags)
{
    m_instance.setFlags(flags);
}

// ------------------------------------------------------------------
uint32_t ShapeInstance::id() const
{
    return m_instance.id();
}

uint32_t ShapeInstance::sbtOffset() const
{
    return m_instance.sbtOffset();
}

uint32_t ShapeInstance::visibilityMask() const
{
    return m_instance.visibilityMask();
}

OptixTraversableHandle ShapeInstance::handle() const
{
    return m_instance.handle();
}

uint32_t ShapeInstance::flags() const
{
    return m_instance.flags();
}

// ------------------------------------------------------------------
void ShapeInstance::setTransform(const Matrix4f& matrix)
{
    m_instance.setTransform(matrix);
}

void ShapeInstance::translate(const Vec3f& t)
{
    m_instance.translate(t);
}

void ShapeInstance::scale(const Vec3f& s)
{
    m_instance.translate(s);
}

void ShapeInstance::scale(const float s)
{
    m_instance.scale(s);
}

void ShapeInstance::rotate(const float radians, const Vec3f& axis)
{
    m_instance.rotate(radians, axis);
}

void ShapeInstance::rotateX(const float radians)
{
    m_instance.rotateX(radians);
}

void ShapeInstance::rotateY(const float radians)
{
    m_instance.rotateY(radians);
}

void ShapeInstance::rotateZ(const float radians)
{
    m_instance.rotateZ(radians);
}

void ShapeInstance::reflect(Axis axis)
{
    m_instance.reflect(axis);
}

Matrix4f ShapeInstance::transform()
{
    return m_instance.transform();
}

// ------------------------------------------------------------------
void ShapeInstance::allowUpdate()
{
    m_gas.allowUpdate();
}

void ShapeInstance::allowCompaction()
{
    m_gas.allowCompaction();
}

void ShapeInstance::preferFastTrace()
{
    m_gas.preferFastTrace();
}

void ShapeInstance::preferFastBuild()
{
    m_gas.preferFastBuild();
}

void ShapeInstance::allowRandomVertexAccess()
{
    m_gas.allowRandomVertexAccess();
}

// ------------------------------------------------------------------
void ShapeInstance::buildAccel(const Context& ctx, CUstream stream)
{
    m_gas.build(ctx, stream);
    m_instance.setTraversableHandle(m_gas.handle());
}

void ShapeInstance::updateAccel(const Context& ctx, CUstream stream)
{
    m_gas.update(ctx, stream);
    m_instance.setTraversableHandle(m_gas.handle());
}

// ------------------------------------------------------------------
OptixInstance* ShapeInstance::rawInstancePtr() const
{
    return m_instance.rawInstancePtr();
}

} // namespace prayground