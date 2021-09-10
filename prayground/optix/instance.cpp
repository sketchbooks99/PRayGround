#include "instance.h"
#include <prayground/core/cudabuffer.h>

namespace prayground {

// ------------------------------------------------------------------
Instance::Instance()
{
    m_instance = new OptixInstance;
    for (int i = 0; i < 12; i++)
        m_instance->transform[i] = i % 5 == 0 ? 1.0f : 0.0f;
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

void Instance::translate(const float3& t)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::translate(t);
    setTransform(current_mat);
}

void Instance::scale(const float3& s)
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

void Instance::rotate(const float radians, const float3& axis)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::rotate(radians, axis);
    setTransform(current_mat);
}

void Instance::rotateX(const float radians)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::rotate(radians, make_float3(1.0f, 0.0f, 0.0f));
    setTransform(current_mat);
}

void Instance::rotateY(const float radians)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::rotate(radians, make_float3(0.0f, 1.0f, 0.0f));
    setTransform(current_mat);
}

void Instance::rotateZ(const float radians)
{
    Matrix4f current_mat(m_instance->transform);
    current_mat *= Matrix4f::rotate(radians, make_float3(0.0f, 0.0f, 1.0f));
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
ShapeInstance::ShapeInstance()
: m_instance(Instance{})
{

}

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

void ShapeInstance::translate(const float3& t)
{
    m_instance.translate(t);
}

void ShapeInstance::scale(const float3& s)
{
    m_instance.translate(s);
}

void ShapeInstance::scale(const float s)
{
    m_instance.scale(s);
}

void ShapeInstance::rotate(const float radians, const float3& axis)
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
    // GASが更新された場合はhandleも更新される？
    m_instance.setTraversableHandle(m_gas.handle());
}

// ------------------------------------------------------------------
OptixInstance* ShapeInstance::rawInstancePtr() const
{
    return m_instance.rawInstancePtr();
}

} // ::prayground