#pragma once

#ifndef __CUDACC__
#include <prayground/core/cudabuffer.h>
#endif

#include <prayground/core/shape.h>

namespace prayground {

class Plane final : public Shape {
public:
    struct Data {
        Vec2f min; 
        Vec2f max;
    };

#ifndef __CUDACC__
    Plane();
    Plane(const Vec2f& min, const Vec2f& max);

    constexpr ShapeType type() override;

    OptixBuildInput createBuildInput() override;

    void copyToDevice() override;
    void free() override;

    AABB bound() const override;

    Data getData() const;
private:
    Vec2f m_min, m_max;
    CUdeviceptr d_aabb_buffer{ 0 };

#endif
};

} // ::prayground