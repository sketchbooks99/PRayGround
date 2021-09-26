#pragma once 

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#endif

namespace prayground {

struct CylinderData
{
    float radius; 
    float height;
};

#ifndef __CUDACC__
class Cylinder final : public Shape {
public:
    using DataType = CylinderData;

    Cylinder();
    Cylinder(float radius, float height);

    ShapeType constexpr type() override;

    void copyToDevice() override;
    void free() override;  

    OptixBuildInput createBuildInput() override;

    AABB bound() const override;

    DataType deviceData() const;

private:
    float m_radius;
    float m_height;
    CUdeviceptr d_aabb_buffer{ 0 };
};
#endif // __CUDACC__

}