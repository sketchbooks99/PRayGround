#pragma once 

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#endif 

namespace prayground {

struct BoxMediumData
{
    float3 min; 
    float3 max; 
    float density;
};

#ifndef __CUDACC__
class BoxMedium final : public Shape {
public:
    using DataType = BoxMediumData;

    BoxMedium();
    BoxMedium(const float3& min, const float3& max, const float density);

    constexpr ShapeType type() override;

    void copyToDevice() override;
    void free() override;

    OptixBuildInput createBuildInput() override;

    AABB bound() const override;

    const float3& min() const;
    const float3& max() const;

    DataType deviceData() const;
private:
    float3 m_min; 
    float3 m_max; 
    float m_density;
    CUdeviceptr d_aabb_buffer{ 0 };
};
#endif

}