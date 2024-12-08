#pragma once

#include <prayground/core/shape.h>

namespace prayground {

    class PointCloud : public Shape {
    public:
        struct Data {
            Vec3f point;
            float radius;
        };

#ifndef __CUDACC__

        PointCloud();
        PointCloud(const std::vector<PointCloud::Data>& points);
        PointCloud(PointCloud::Data* points, uint32_t num_points);

        constexpr ShapeType type() override;
        OptixBuildInput createBuildInput() override;

        uint32_t numPrimitives() const override;

        void copyToDevice() override;

        AABB bound() const override;

        void updatePoints(PointCloud::Data* points, uint32_t num_points);
        void updatePoints(const std::vector<PointCloud::Data>& points);

        /* Getter of host-side pointer for points */
        const PointCloud::Data* points();
    private:
        std::unique_ptr<Data[]> m_points;
        CUdeviceptr d_points{ 0 };

        uint32_t m_num_points;

        CUdeviceptr d_aabb_buffer{ 0 };
#endif
    };

} // namespace prayground