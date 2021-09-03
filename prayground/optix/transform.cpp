#include "transform.h"
#include <prayground/core/util.h>

namespace prayground {

// ---------------------------------------------------------------------------
Transform::Transform(TransformType type)
: m_type(type)
{

}

Transform::Transform(const OptixStaticTransform& static_transform)
{
    m_type = TransformType::Static;
    m_transform = new OptixStaticTransform(static_transform);
}

Transform::Transform(const OptixMatrixMotionTransform& matrix_motion_transform)
{
    m_type = TransformType::MatrixMotion;
    m_transform = new OptixMatrixMotionTransform(matrix_motion_transform);
}

Transform::Transform(const OptixSRTMotionTransform& srt_motion_transform)
{
    m_type = TransformType::SRTMotion;
    m_transform = new OptixSRTMotionTransform(srt_motion_transform);
}


// ---------------------------------------------------------------------------
void Transform::setStaticTransform(const Matrix4f& m)
{
    Assert(m_type == TransformType::Static, 
        "prayground::Transform::setStaticTransform(): The type of must be a static transform");
    
    Matrix4f inv = m.inverse();
    OptixStaticTransform* static_transform = std::get<OptixStaticTransform*>(m_transform);
    memcpy(static_transform->transform, m.data(), sizeof(float)*12);
    memcpy(static_transform->invTransform, inv.data(), sizeof(float)*12);
}

void Transform::setStaticTransform(const float m[12])
{
    Assert(m_type == TransformType::Static, 
        "prayground::Transform::setStaticTransform(): The type of must be a static transform");
    Matrix4f inv = Matrix4f(m).inverse();
    OptixStaticTransform* static_transform = std::get<OptixStaticTransform*>(m_transform);
    memcpy(static_transform->transform, m, sizeof(float)*12);
    memcpy(static_transform->invTransform, inv.data(), sizeof(float)*12);
}

// ---------------------------------------------------------------------------
void Transform::setMatrixMotionTransform(const Matrix4f& m, uint32_t idx)
{
    Assert(m_type == TransformType::MatrixMotion, 
        "prayground::Transform::setMatrixMotionTransform(): The type of must be a matrix motion transform");
    Assert(idx >= 0 && idx < 2,
        "prayground::Transform::setMatrixMotionTransform(): The index of transform must be 0 or 1.");

    OptixMatrixMotionTransform* matrix_motion_transform = std::get<OptixMatrixMotionTransform*>(m_transform);
    memcpy(matrix_motion_transform->transform[idx], m.data(), sizeof(float)*12);
}

void Transform::setMatrixMotionTransform(const float m[12], uint32_t idx)
{
    Assert(m_type == TransformType::MatrixMotion, 
        "prayground::Transform::setMatrixMotionTransform(): The type of must be a matrix motion transform");
    Assert(idx >= 0 && idx < 2,
        "prayground::Transform::setMatrixMotionTransform(): The index of transform must be 0 or 1.");
    
    OptixMatrixMotionTransform* matrix_motion_transform = std::get<OptixMatrixMotionTransform*>(m_transform);
    memcpy(matrix_motion_transform->transform[idx], m, sizeof(float)*12);
}

// ---------------------------------------------------------------------------
void Transform::setSRTMotionTransform(const OptixSRTData& srt_data, uint32_t idx)
{
    Assert(m_type == TransformType::SRTMotion, 
        "prayground::Transform::setSRTMotionTransform(): The type of must be a SRT motion transform");

    Assert(idx >= 0 && idx < 2,
        "prayground::Transform::setSRTMotionTransform(): The index of transform must be 0 or 1.");

    OptixSRTMotionTransform* srt_motion_transform = std::get<OptixSRTMotionTransform*>(m_transform);
    srt_motion_transform->srtData[idx] = srt_data;
}

// ---------------------------------------------------------------------------
void Transform::setNumKey(uint16_t num_key)
{
    Assert(m_type == TransformType::MatrixMotion || m_type == TransformType::SRTMotion, 
        "prayground::Transform::setNumKey(): Static transform could not determine a motion option.");
    
    if (m_type == TransformType::MatrixMotion)
    {
        OptixMatrixMotionTransform* matrix_motion = std::get<OptixMatrixMotionTransform*>(m_transform);
        matrix_motion->motionOptions.numKeys = num_key;
    }
    else if (m_type == TransformType::SRTMotion)
    {
        OptixSRTMotionTransform* srt_motion = std::get<OptixSRTMotionTransform*>(m_transform);
        srt_motion->motionOptions.numKeys = num_key;
    }
}

// ---------------------------------------------------------------------------
void Transform::enableStartVanish()
{
    Assert(m_type == TransformType::MatrixMotion || m_type == TransformType::SRTMotion, 
        "prayground::Transform::enableStartVanish(): Static transform could not determine a motion option.");

    if (m_type == TransformType::MatrixMotion)
    {
        OptixMatrixMotionTransform* matrix_motion = std::get<OptixMatrixMotionTransform*>(m_transform);
        matrix_motion->motionOptions.flags |= OPTIX_MOTION_FLAG_START_VANISH;
    }
    else if (m_type == TransformType::SRTMotion)
    {
        OptixSRTMotionTransform* srt_motion = std::get<OptixSRTMotionTransform*>(m_transform);
        srt_motion->motionOptions.flags |= OPTIX_MOTION_FLAG_START_VANISH;
    }
}

void Transform::disableStartVanish()
{
    Assert(m_type == TransformType::MatrixMotion || m_type == TransformType::SRTMotion, 
        "prayground::Transform::disableStartVanish(): Static transform could not determine a motion option.");

    if (m_type == TransformType::MatrixMotion)
    {
        OptixMatrixMotionTransform* matrix_motion = std::get<OptixMatrixMotionTransform*>(m_transform);
        matrix_motion->motionOptions.flags &= ~OPTIX_MOTION_FLAG_START_VANISH;
    }
    else if (m_type == TransformType::SRTMotion)
    {
        OptixSRTMotionTransform* srt_motion = std::get<OptixSRTMotionTransform*>(m_transform);
        srt_motion->motionOptions.flags &= ~OPTIX_MOTION_FLAG_START_VANISH;
    }
}

void Transform::enableEndVanish()
{
    Assert(m_type == TransformType::MatrixMotion || m_type == TransformType::SRTMotion, 
        "prayground::Transform::enableEndVanish(): Static transform could not determine a motion option.");

    if (m_type == TransformType::MatrixMotion)
    {
        OptixMatrixMotionTransform* matrix_motion = std::get<OptixMatrixMotionTransform*>(m_transform);
        matrix_motion->motionOptions.flags |= OPTIX_MOTION_FLAG_END_VANISH;
    }
    else if (m_type == TransformType::SRTMotion)
    {
        OptixSRTMotionTransform* srt_motion = std::get<OptixSRTMotionTransform*>(m_transform);
        srt_motion->motionOptions.flags |= OPTIX_MOTION_FLAG_END_VANISH;
    }
}

void Transform::disableEndVanish()
{
    Assert(m_type == TransformType::MatrixMotion || m_type == TransformType::SRTMotion, 
        "prayground::Transform::disableEndVanish(): Static transform could not determine a motion option.");
    

    if (m_type == TransformType::MatrixMotion)
    {
        OptixMatrixMotionTransform* matrix_motion = std::get<OptixMatrixMotionTransform*>(m_transform);
        matrix_motion->motionOptions.flags &= ~OPTIX_MOTION_FLAG_END_VANISH;
    }
    else if (m_type == TransformType::SRTMotion)
    {
        OptixSRTMotionTransform* srt_motion = std::get<OptixSRTMotionTransform*>(m_transform);
        srt_motion->motionOptions.flags &= ~OPTIX_MOTION_FLAG_END_VANISH;
    }
}

// ---------------------------------------------------------------------------
void Transform::setBeginTime(float start_time)
{
    Assert(m_type == TransformType::MatrixMotion || m_type == TransformType::SRTMotion, 
        "prayground::Transform::setBeginTime(): Static transform could not determine a motion time.");

    if (m_type == TransformType::MatrixMotion)
    {
        OptixMatrixMotionTransform* matrix_motion = std::get<OptixMatrixMotionTransform*>(m_transform);
        matrix_motion->motionOptions.timeBegin = start_time;
    }
    else if (m_type == TransformType::SRTMotion)
    {
        OptixSRTMotionTransform* srt_motion = std::get<OptixSRTMotionTransform*>(m_transform);
        srt_motion->motionOptions.timeBegin = start_time;
    }
}

void Transform::setEndTime(float end_time)
{
    Assert(m_type == TransformType::MatrixMotion || m_type == TransformType::SRTMotion, 
        "prayground::Transform::setEndTime(): Static transform could not determine a motion time.");

    if (m_type == TransformType::MatrixMotion)
    {
        OptixMatrixMotionTransform* matrix_motion = std::get<OptixMatrixMotionTransform*>(m_transform);
        matrix_motion->motionOptions.timeEnd = end_time;
    }
    else if (m_type == TransformType::SRTMotion)
    {
        OptixSRTMotionTransform* srt_motion = std::get<OptixSRTMotionTransform*>(m_transform);
        srt_motion->motionOptions.timeEnd = end_time;
    }
}

// ---------------------------------------------------------------------------
void Transform::setChildHandle(OptixTraversableHandle handle)
{
    std::visit([&](auto transform) {
        transform->child = handle;
    }, m_transform);
}

// ---------------------------------------------------------------------------
OptixTraversableHandle Transform::handle() const 
{
    return std::visit([](auto transform) {
        return transform->child;
    }, m_transform);
}

TransformType Transform::type() const 
{
    return m_type;
}

} // ::prayground