#pragma once

#include <prayground/core/shape.h>

namespace prayground {

    class PointCloud : public Shape {
    public:
        struct Data {
            Vec3f* points;
            uint32_t num_points;
            float radius;
        };

#ifndef __CUDACC__

        PointCloud();
        PointCloud(Vec3f* points, uint32_t num_points, float radius);
        PointCloud(const std::vector<Vec3f>& points, float radius);

        constexpr ShapeType type() override;
        OptixBuildInput createBuildInput() override;

        uint32_t numPrimitives() const override;

        void copyToDevice() override;

        AABB bound() const override;

        Data getData() const;

        void updatePoints(Vec3f* points, uint32_t num_points);

        /* Getter of host-side pointer for points */
        const Vec3f* points();

        /* Getter of device-side pointer for points */
        Vec3f* devicePoints();
    private:
        std::unique_ptr<Vec3f> m_points;
        CUdeviceptr d_points{ 0 };

        uint32_t m_num_points;

        float m_radius;

        CUdeviceptr d_aabb_buffer{ 0 };
#endif
    };

} // namespace prayground