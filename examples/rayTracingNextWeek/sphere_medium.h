#pragma once 

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#endif 

namespace prayground {

struct SphereMediumData
{
    Vec3f center;
    float radius; 
    float density;
};

#ifndef __CUDACC__
class SphereMedium final : public Shape {
public:
    using DataType = SphereMediumData;

    SphereMedium();
    SphereMedium(const Vec3f& center, const float radius, const float density);

    constexpr ShapeType type() override;

    void copyToDevice() override;
    void free() override;

    OptixBuildInput createBuildInput() override;

    AABB bound() const override;

    const Vec3f& center() const;
    const float& radius() const;

    DataType deviceData() const;
private:
    Vec3f m_center;
    float m_radius; 
    float m_density;
    CUdeviceptr d_aabb_buffer{ 0 };
};
#endif

}

