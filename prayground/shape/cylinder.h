#pragma once 

#include <prayground/core/shape.h>

namespace prayground {

class Cylinder final : public Shape {
public:
    struct Data {
        float radius;
        float height;
    };

#ifndef __CUDACC__
    Cylinder();
    Cylinder(float radius, float height);

    constexpr ShapeType type() override;

    void copyToDevice() override;
    void free() override;  

    OptixBuildInput createBuildInput() override;

    AABB bound() const override;

    Data getData() const;

private:
    float m_radius;
    float m_height;
    CUdeviceptr d_aabb_buffer{ 0 };

#endif
};

} // ::prayground