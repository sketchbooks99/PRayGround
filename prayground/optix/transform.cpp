#include "transform.h"
#include <prayground/core/util.h>

namespace prayground {

// ---------------------------------------------------------------------------
Transform::Transform(TransformType type)
: m_type(type)
{
    switch(m_type)
    {
        case TransformType::Static:
            m_transform = new OptixStaticTransform;
            break;
        case TransformType::MatrixMotion:
            m_transform = new OptixMatrixMotionTransform;
            break;
        case TransformType::SRTMotion:
            m_transform = new OptixSRTMotionTransform;
            break;
    }
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

void Transform::setMatrixMotionTransform(const Matrix4f& m1, const Matrix4f& m2)
{
    setMatrixMotionTransform(m1, 0);
    setMatrixMotionTransform(m2, 1);
}

void Transform::setMatrixMotionTransform(const float m1[12], const float m2[12])
{
    setMatrixMotionTransform(m1, (uint32_t)0);
    setMatrixMotionTransform(m2, (uint32_t)1);
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

void Transform::setSRTMotionTransform(const OptixSRTData& srt1, const OptixSRTData& srt2)
{
    setSRTMotionTransform(srt1, (uint32_t)0);
    setSRTMotionTransform(srt2, (uint32_t)1);
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
        "prayground::Transform::setBeginTime(): The begin time could be set to static transform.");

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
        "prayground::Transform::setEndTime(): The end time could be set to static transform.");

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
void Transform::setMotionOptions(const OptixMotionOptions& options)
{
    Assert(m_type == TransformType::MatrixMotion || m_type == TransformType::SRTMotion,
        "prayground::Transform::setEndTime(): Motion option could not be set to static transform.");
    
    if (m_type == TransformType::MatrixMotion)
    {
        OptixMatrixMotionTransform* matrix_motion = std::get<OptixMatrixMotionTransform*>(m_transform);
        matrix_motion->motionOptions = options;
    }
    else if (m_type == TransformType::SRTMotion)
    {
        OptixSRTMotionTransform* srt_motion = std::get<OptixSRTMotionTransform*>(m_transform);
        srt_motion->motionOptions = options;
    }
}

// ---------------------------------------------------------------------------
void Transform::setChildHandle(OptixTraversableHandle handle)
{
    std::visit([&](auto transform) {
        transform->child = handle;
    }, m_transform);
}

OptixTraversableHandle Transform::childHandle() const 
{
    return std::visit([](auto transform) {
        return transform->child;
    }, m_transform);
}

// ---------------------------------------------------------------------------
TransformType Transform::type() const 
{
    return m_type;
}

// ---------------------------------------------------------------------------
void Transform::buildHandle(const Context& ctx)
{
    switch (m_type)
    {
        case TransformType::Static:
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_transform),
                sizeof(OptixStaticTransform)
            ));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_transform),
                std::get<OptixStaticTransform*>(m_transform),
                sizeof(OptixStaticTransform),
                cudaMemcpyHostToDevice
            ));
            break;
        case TransformType::MatrixMotion:
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_transform),
                sizeof(OptixMatrixMotionTransform)
            ));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_transform),
                std::get<OptixMatrixMotionTransform*>(m_transform),
                sizeof(OptixMatrixMotionTransform),
                cudaMemcpyHostToDevice
            ));
            break;
        case TransformType::SRTMotion:
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_transform),
                sizeof(OptixSRTMotionTransform)
            ));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_transform),
                std::get<OptixSRTMotionTransform*>(m_transform),
                sizeof(OptixSRTMotionTransform),
                cudaMemcpyHostToDevice
            ));
            break;
        default:
            break;
    }
    OPTIX_CHECK(optixConvertPointerToTraversableHandle(
        static_cast<OptixDeviceContext>(ctx), 
        d_transform, 
        static_cast<OptixTraversableType>(m_type), 
        &m_handle
    ));
}

OptixTraversableHandle Transform::handle() const 
{
    return m_handle;
}


} // ::prayground