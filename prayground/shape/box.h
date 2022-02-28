#pragma once 

#include <prayground/core/shape.h>

namespace prayground {

struct BoxData 
{
    float3 min;
    float3 max;
};

class Box final : public Shape {
public:
    struct Data {
        float3 min; 
        float3 max;
    };

#ifndef __CUDACC__
    Box();
    Box(const float3& min, const float3& max);

    constexpr ShapeType type() override;

    void copyToDevice() override;
    void free() override;

    OptixBuildInput createBuildInput() override;

    AABB bound() const override;

    const float3& min() const;
    const float3& max() const;

    Data getData() const;
private:
    float3 m_min;
    float3 m_max;
    CUdeviceptr d_aabb_buffer{ 0 };

#endif
};

} // ::prayground