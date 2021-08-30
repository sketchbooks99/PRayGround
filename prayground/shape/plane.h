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
    Plane(const float2& min, const float2& max);
    OptixBuildInputType buildInputType() const override;

    void copyToDevice() override;
    
    void buildInput( OptixBuildInput& bi ) override;
    AABB bound() const override;
private:
    float2 m_min, m_max;
};
#endif // __CUDACC__

} // ::prayground