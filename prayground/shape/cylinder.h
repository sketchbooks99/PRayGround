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
    Cylinder();
    Cylinder(float radius, float height);

    OptixBuildInputType buildInputType() const override;

    void copyToDevice() override;
    OptixBuildInput createBuildInput() override;
    void free() override;  

    AABB bound() const;

private:
    float m_radius;
    float m_height;
    CUdeviceptr d_aabb_buffer{ 0 };
};
#endif // __CUDACC__

}