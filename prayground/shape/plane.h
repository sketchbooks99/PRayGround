#pragma once

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#include <prayground/core/cudabuffer.h>
#endif

namespace prayground {

class Plane final : public Shape {
public:
    struct Data {
        float2 min; 
        float2 max;
    };

#ifndef __CUDACC__
    Plane();
    Plane(const float2& min, const float2& max);

    constexpr ShapeType type() override;

    OptixBuildInput createBuildInput() override;

    void copyToDevice() override;
    void free() override;

    AABB bound() const override;

    Data getData() const;
private:
    float2 m_min, m_max;
    CUdeviceptr d_aabb_buffer{ 0 };

#endif
};

} // ::prayground