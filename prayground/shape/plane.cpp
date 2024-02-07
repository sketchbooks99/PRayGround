#include "plane.h"
#include <prayground/core/util.h>

namespace prayground {

    // ------------------------------------------------------------------
    Plane::Plane()
    : m_min{-1.0f, -1.0f}, m_max{1.0f, 1.0f}
    {

    }

    Plane::Plane(const Vec2f& min, const Vec2f& max)
    : m_min{ min }, m_max{ max }
    {

    }

    // ------------------------------------------------------------------
    constexpr ShapeType Plane::type()
    {
        return ShapeType::Custom;
    }

    // ------------------------------------------------------------------
    void Plane::copyToDevice() 
    {
        Data data = this->getData();

        if (!d_data)
            CUDA_CHECK( cudaMalloc( &d_data, sizeof(Data) ) );
        CUDA_CHECK( cudaMemcpy(
            d_data, 
            &data, sizeof(Data), 
            cudaMemcpyHostToDevice
        ));
    }

    // ------------------------------------------------------------------
    OptixBuildInput Plane::createBuildInput()
    {
        if (d_aabb_buffer) cuda_free(d_aabb_buffer);
        return createSingleCustomBuildInput(d_aabb_buffer, this->bound(), m_sbt_index);
    }

    uint32_t Plane::numPrimitives() const
    {
        return 1u;
    }

    // ------------------------------------------------------------------
    void Plane::free()
    {
        Shape::free();
        if (d_aabb_buffer) {
            cuda_free(d_aabb_buffer);
        }
        d_aabb_buffer = 0;
    }

    // ------------------------------------------------------------------
    AABB Plane::bound() const 
    {
        AABB box{Vec3f(m_min.x(), -0.01f, m_min.y()), Vec3f(m_max.x(), 0.01f, m_max.y())};
        return box;
    }

    // ------------------------------------------------------------------
    Plane::Data Plane::getData() const 
    {
        return { m_min, m_max };
    }

} // namespace prayground