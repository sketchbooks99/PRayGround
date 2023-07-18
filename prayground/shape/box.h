#pragma once 

#include <prayground/core/shape.h>

namespace prayground {

class Box final : public Shape {
public:
    struct Data {
        Vec3f min; 
        Vec3f max;
    };

#ifndef __CUDACC__
    Box();
    explicit Box(const Vec3f& min, const Vec3f& max);

    constexpr ShapeType type() override;

    void copyToDevice() override;
    void free() override;

    OptixBuildInput createBuildInput() override;

    uint32_t numPrimitives() const override;

    AABB bound() const override;

    const Vec3f& min() const;
    const Vec3f& max() const;

    Data getData() const;
private:
    Vec3f m_min;
    Vec3f m_max;
    CUdeviceptr d_aabb_buffer{ 0 };

#endif
};

} // ::prayground