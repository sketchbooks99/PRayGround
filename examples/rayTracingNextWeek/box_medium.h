#pragma once 

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#endif 

namespace prayground {

struct BoxMediumData
{
    Vec3f min; 
    Vec3f max; 
    float density;
};

#ifndef __CUDACC__
class BoxMedium final : public Shape {
public:
    using DataType = BoxMediumData;

    BoxMedium();
    BoxMedium(const Vec3f& min, const Vec3f& max, const float density);

    constexpr ShapeType type() override;

    void copyToDevice() override;
    void free() override;

    OptixBuildInput createBuildInput() override;

    AABB bound() const override;

    const Vec3f& min() const;
    const Vec3f& max() const;

    DataType deviceData() const;
private:
    Vec3f m_min; 
    Vec3f m_max; 
    float m_density;
    CUdeviceptr d_aabb_buffer{ 0 };
};
#endif

}