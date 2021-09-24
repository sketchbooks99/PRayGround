#pragma once 

#include <optix.h>
#include <prayground/math/matrix.h>
#include <prayground/optix/context.h>
#include <variant>

namespace prayground {

enum class TransformType
{
    Static = OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM,
    MatrixMotion = OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
    SRTMotion = OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM
};

class Transform {
public:
    Transform() = default;
    Transform(TransformType type);
    Transform(const OptixStaticTransform& static_transform);
    Transform(const OptixMatrixMotionTransform& matrix_motion_transform);
    Transform(const OptixSRTMotionTransform& srt_motion_transform);
    
    void setStaticTransform(const Matrix4f& m);
    void setStaticTransform(const float m[12]);
    
    void setMatrixMotionTransform(const Matrix4f& m, uint32_t idx);
    void setMatrixMotionTransform(const float m[12], uint32_t idx);
    void setMatrixMotionTransform(const Matrix4f& m1, const Matrix4f& m2);
    void setMatrixMotionTransform(const float m1[12], const float m2[12]);

    void setSRTMotionTransform(const OptixSRTData& srt_data, uint32_t idx);
    void setSRTMotionTransform(const OptixSRTData& srt1, const OptixSRTData& srt2);

    void setNumKey(uint16_t num_key);
    
    void enableStartVanish();
    void disableStartVanish();
    void enableEndVanish();
    void disableEndVanish();

    void setBeginTime(float start_time);
    void setEndTime(float end_time);

    void setMotionOptions(const OptixMotionOptions& options);

    void setChildHandle(OptixTraversableHandle handle);
    OptixTraversableHandle childHandle() const;

    void buildHandle(const Context& ctx);
    OptixTraversableHandle handle() const;

    TransformType type() const;
private:
    TransformType m_type;
    std::variant<OptixStaticTransform*, OptixMatrixMotionTransform*, OptixSRTMotionTransform*> m_transform; 
    OptixTraversableHandle m_handle;
    CUdeviceptr d_transform;
};

} // ::prayground