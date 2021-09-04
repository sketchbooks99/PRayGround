#pragma once

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#include <prayground/core/cudabuffer.h>
#endif

namespace prayground {

struct PlaneData 
{
    float2 min;
    float2 max;
};

#ifndef __CUDACC__
class Plane final : public Shape {
public:
    using DataType = PlaneData;

    Plane();
    Plane(const float2& min, const float2& max);

    ShapeType type() const override;

    void copyToDevice() override;
    OptixBuildInput createBuildInput() override;

    void free() override;

    AABB bound() const override;
private:
    float2 m_min, m_max;
    CUdeviceptr d_aabb_buffer{ 0 };
};
#endif // __CUDACC__

} // ::prayground