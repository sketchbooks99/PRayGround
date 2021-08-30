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
    Cylinder(float radius, float height);

    OptixBuildInputType buildInputType() const override;

    void copyToDevice() override;

    void buildInput( OptixBuildInput& bi ) override;

    AABB bound() const override;

private:
    float m_radius;
    float m_height;
};
#endif // __CUDACC__

}