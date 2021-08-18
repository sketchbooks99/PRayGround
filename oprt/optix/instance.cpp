#include "instance.h"
#include <oprt/core/cudabuffer.h>

namespace oprt {

// ------------------------------------------------------------------
Instance::Instance()
{
    
}

Instance::Instance(const Transform& transform)
{
    setTransform(transform);
}

// ------------------------------------------------------------------
void Instance::copyToDevice()
{
    CUDABuffer<OptixInstance> d_instance_buffer;
    d_instance_buffer.copyToDevice(&m_instance, sizeof(OptixInstance));
    d_instance = d_instance_buffer.devicePtr();
}

// ------------------------------------------------------------------
void Instance::setId(const uint32_t id)
{
    m_instance.instanceId = id;
}

void Instance::setSBTOffset(const uint32_t sbt_offset)
{
    m_instance.sbtOffset = sbt_offset;
}

void Instance::setVisibilityMask(const uint32_t visibility_mask)
{
    m_instance.visibilityMask = visibility_mask;
}

void Instance::setTraversableHandle(OptixTraversableHandle handle)
{
    m_instance.traversableHandle = handle;
}

void Instance::setPadding(uint32_t pad[2])
{
    memcpy(m_instance.pad, pad, sizeof(uint32_t) * 2);
}

void Instance::setFlags(const uint32_t flags)
{
    m_instance.flags = flags;
}

// ------------------------------------------------------------------
void Instance::setTransform(const Transform& transform)
{
    memcpy(m_instance.transform, transform.mat.data(), sizeof(float)*12);
}

void Instance::setTransform(const Matrix4f& matrix)
{
    memcpy(m_instance.transform, matrix.data(), sizeof(float)*12);
}

void Instance::translate(const float3& t)
{
    Matrix4f current_mat(m_instance.transform);
    current_mat *= Matrix4f::translate(t);
    setTransform(current_mat);
}

void Instance::scale(const float3& s)
{
    Matrix4f current_mat(m_instance.transform);
    current_mat *= Matrix4f::scale(s);
    setTransform(current_mat);
}

void Instance::scale(const float s)
{
    Matrix4f current_mat(m_instance.transform);
    current_mat *= Matrix4f::scale(s);
    setTransform(current_mat);
}

void Instance::rotate(const float radians, const float3& axis)
{
    Matrix4f current_mat(m_instance.transform);
    current_mat *= Matrix4f::rotate(radians, axis);
    setTransform(current_mat);
}

void Instance::rotateX(const float radians)
{
    Matrix4f current_mat(m_instance.transform);
    current_mat *= Matrix4f::rotate(radians, make_float3(1.0f, 0.0f, 0.0f));
    setTransform(current_mat);
}

void Instance::rotateY(const float radians)
{
    Matrix4f current_mat(m_instance.transform);
    current_mat *= Matrix4f::rotate(radians, make_float3(0.0f, 1.0f, 0.0f));
    setTransform(current_mat);
}

void Instance::rotateZ(const float radians)
{
    Matrix4f current_mat(m_instance.transform);
    current_mat *= Matrix4f::rotate(radians, make_float3(0.0f, 0.0f, 1.0f));
    setTransform(current_mat);
}

Transform Instance::transform() const 
{
    return Transform(m_instance.transform);
}

// ------------------------------------------------------------------
bool Instance::isDataOnDevice() const 
{
    return static_cast<bool>(d_instance);
}

CUdeviceptr Instance::devicePtr() const 
{
    return d_instance;
}

} // ::oprt