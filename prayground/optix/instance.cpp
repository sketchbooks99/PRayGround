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
}

Instance::Instance(const Matrix4f& matrix)
{
    m_instance = new OptixInstance;
    setTransform(matrix);
    m_instance->flags = OPTIX_INSTANCE_FLAG_NONE;
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
OptixTraversableHandle Instance::handle()
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

} // ::prayground