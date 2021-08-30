#pragma once 

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#endif

namespace prayground {

struct SphereData {
    float3 center;
    float radius;
};

#ifndef __CUDACC__
class Sphere final : public Shape {
public:
    Sphere(const float3& c, float r);

    OptixBuildInputType buildInputType() const override;

    void copyToDevice() override;
    void buildInput( OptixBuildInput& bi ) override;

    AABB bound() const override;
private:
    float3 m_center;
    float m_radius;
};

#endif

}