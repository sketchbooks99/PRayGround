#pragma once 

#include <prayground/core/shape.h>

namespace prayground {

    class SphereMedium final : public Shape {
    public:
        struct Data {
            Vec3f center; 
            float radius; 
            float density;
        };

#ifndef __CUDACC__
        SphereMedium();
        SphereMedium(const Vec3f& center, const float radius, const float density);

        constexpr ShapeType type() override;

        void copyToDevice() override;
        void free() override;

        OptixBuildInput createBuildInput() override;

        AABB bound() const override;

        const Vec3f& center() const;
        const float& radius() const;

        Data getData() const;
    private:
        Vec3f m_center;
        float m_radius; 
        float m_density;
        CUdeviceptr d_aabb_buffer{ 0 };
#endif
    };

} // namespace prayground

