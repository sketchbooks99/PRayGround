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
        return OptixBuildInput();
    }

    // ------------------------------------------------------------------
    uint32_t PointCloud::numPrimitives() const
    {
        return m_num_points;
    }

    // ------------------------------------------------------------------
    void PointCloud::copyToDevice()
    {
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
        return Data();
    }

    // ------------------------------------------------------------------
    void PointCloud::updatePoints(const Vec3f* points)
    {
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
