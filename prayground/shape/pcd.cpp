#include "pcd.h"

namespace prayground {

    // ------------------------------------------------------------------
    PointCloud::PointCloud()
    {
        m_points = std::make_unique<Data[]>(0);
        m_num_points = 0;
    }

    // ------------------------------------------------------------------
    PointCloud::PointCloud(const std::vector<PointCloud::Data>& points)
    {
        m_points = std::make_unique<Data[]>(points.size());
        memcpy(m_points.get(), points.data(), sizeof(Data) * points.size());
        m_num_points = static_cast<uint32_t>(points.size());
    }

    // ------------------------------------------------------------------
    PointCloud::PointCloud(PointCloud::Data* points, uint32_t num_points)
    {
        m_points = std::make_unique<Data[]>(m_num_points);
        memcpy(m_points.get(), points, sizeof(Data) * num_points);
        m_num_points = num_points;
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
        std::vector<uint32_t> sbt_indices;
        uint32_t* input_flags = new uint32_t[m_num_points];
        for (int i = 0; i < m_num_points; i++) {
            input_flags[i] = OPTIX_GEOMETRY_FLAG_NONE;
            sbt_indices.push_back(m_sbt_index);
        }

        CUDABuffer<uint32_t> d_sbt_indices;
        d_sbt_indices.copyToDevice(sbt_indices);

        // Create AABB buffer
        std::vector<OptixAabb> aabbs(m_num_points);
        for (int i = 0; i < m_num_points; i++) {
            PointCloud::Data data = m_points.get()[i];
            aabbs[i] = static_cast<OptixAabb>(AABB(data.point - data.radius, data.point + data.radius));
        }
        CUDABuffer<OptixAabb> d_aabbs;
        d_aabbs.copyToDevice(aabbs);
        d_aabb_buffer = d_aabbs.devicePtr();

        build_input.type = static_cast<OptixBuildInputType>(this->type());
        build_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
        build_input.customPrimitiveArray.numPrimitives = static_cast<uint32_t>(m_num_points);
        build_input.customPrimitiveArray.flags = input_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1u;
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
        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data) * m_num_points));
        CUDA_CHECK(cudaMemcpy(d_data, m_points.get(), sizeof(Data) * m_num_points, cudaMemcpyHostToDevice));
    }

    // ------------------------------------------------------------------
    AABB PointCloud::bound() const
    {
        /* NOTE: Should I aggreagte all points into single AABB? */
        AABB aabb;
        for (uint32_t i = 0; i < m_num_points; i++) {
            auto p = m_points.get()[i];
            aabb = AABB::merge(aabb, AABB(p.point - p.radius, p.point + p.radius));
        }
        return aabb;
    }

    // ------------------------------------------------------------------
    void PointCloud::updatePoints(PointCloud::Data* points, uint32_t num_points)
    {
        m_points.reset(points);
        m_num_points = num_points;
    }

    void PointCloud::updatePoints(const std::vector<PointCloud::Data>& points)
    {
        if (!m_points)
            m_points = std::make_unique<Data[]>(points.size());
        memcpy(m_points.get(), points.data(), sizeof(Data) * points.size());
        m_num_points = static_cast<uint32_t>(points.size());
    }

    // ------------------------------------------------------------------
    const PointCloud::Data* PointCloud::points()
    {
        return m_points.get();
    }

} // namespace prayground
