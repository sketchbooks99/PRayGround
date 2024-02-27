#include "pcd.h"

namespace prayground {

    // ------------------------------------------------------------------
    PointCloud::PointCloud()
    {
        m_points = std::make_unique<Vec3f>();
        m_num_points = 0;
        m_radius = 1.0f;
    }

    // ------------------------------------------------------------------
    PointCloud::PointCloud(Vec3f* points, uint32_t num_points, float radius)
        : m_num_points{num_points}, m_radius{ radius }
    {
        m_points = std::make_unique<Vec3f>(num_points);
        memcpy(m_points.get(), points, sizeof(Vec3f) * num_points);
    }

    // ------------------------------------------------------------------
    PointCloud::PointCloud(const std::vector<Vec3f>& points, float radius)
        : m_radius{radius}
    {
        m_num_points = static_cast<uint32_t>(points.size());
        m_points = std::make_unique<Vec3f>(m_num_points);
        memcpy(m_points.get(), points.data(), sizeof(Vec3f) * m_num_points);
    }

    // ------------------------------------------------------------------
    constexpr ShapeType PointCloud::type()
    {
        return ShapeType::Custom;
    }

    // ------------------------------------------------------------------
    OptixBuildInput PointCloud::createBuildInput()
    {
        OptixBuildInput build_input = {};

        // Copy SBT indices to device
        CUDABuffer<uint32_t> d_sbt_indices;
        uint32_t* sbt_indices = new uint32_t[1];
        sbt_indices[0] = m_sbt_index;
        d_sbt_indices.copyToDevice(sbt_indices, sizeof(uint32_t));

        // Initialize input flags
        std::vector<uint32_t> input_flags(m_num_points, OPTIX_GEOMETRY_FLAG_NONE);

        // Create AABB buffer
        std::vector<OptixAabb> aabbs(m_num_points);
        for (int i = 0; i < m_num_points; i++) 
            aabbs[i] = static_cast<OptixAabb>(AABB(m_points[i] - Vec3f(m_radius), m_points[i] + Vec3f(m_radius)));
        CUDABuffer<OptixAabb> d_aabbs;
        d_aabbs.copyToDevice(aabbs);

        build_input.type = static_cast<OptixBuildInputType>(this->type());
        build_input.customPrimitiveArray.aabbBuffers = &d_aabbs.devicePtr();
        build_input.customPrimitiveArray.numPrimitives = m_num_points;
        build_input.customPrimitiveArray.flags = input_flags.data();
        build_input.customPrimitiveArray.numSbtRecords = 1;
        build_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_indices.devicePtr();
        build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        build_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

        return build_input;
    }

    // ------------------------------------------------------------------
    uint32_t PointCloud::numPrimitives() const
    {
        return m_num_points;
    }

    // ------------------------------------------------------------------
    void PointCloud::copyToDevice()
    {
        Data data = this->getData();

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(Data), cudaMemcpyHostToDevice));
    }

    // ------------------------------------------------------------------
    AABB PointCloud::bound() const
    {
        /* NOTE: Should I aggreagte all points into single AABB? */
        return AABB();
    }

    // ------------------------------------------------------------------
    PointCloud::Data PointCloud::getData() const
    {
        CUDABuffer<Vec3f> d_points;
        d_points.copyToDevice(m_points.get(), m_num_points);
        Data data = { d_points.deviceData(), m_num_points, m_radius };

        return data;
    }

    // ------------------------------------------------------------------
    void PointCloud::updatePoints(Vec3f* points, uint32_t num_points)
    {
        m_points.reset(points);
        m_num_points = num_points;
    }

    // ------------------------------------------------------------------
    const Vec3f* PointCloud::points()
    {
        return m_points.get();
    }

    // ------------------------------------------------------------------
    Vec3f* PointCloud::devicePoints()
    {
        return reinterpret_cast<Vec3f*>(d_points);
    }

} // namespace prayground
