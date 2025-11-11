#pragma once 

#include <prayground/core/shape.h>

namespace prayground {

class Sphere final : public Shape {
public:
    struct Data {
        Vec3f center;
        float radius;
    };

#ifndef __CUDACC__
    Sphere();
    explicit Sphere(float r);
    explicit Sphere(const Vec3f& c, float r);

    constexpr ShapeType type() override;
    OptixBuildInput createBuildInput() override;

    void setCenter(const Vec3f& c) { m_center = c; }
    const Vec3f& center() const { return m_center; }

    void setRadius(float r) { m_radius = r; }
    float radius() const { return m_radius; }

    uint32_t numPrimitives() const override;

    void copyToDevice() override;

    AABB bound() const override;

    Data getData() const;
private:
    Vec3f m_center;
    float m_radius;
    CUdeviceptr d_aabb_buffer{ 0 };

#endif
};

}