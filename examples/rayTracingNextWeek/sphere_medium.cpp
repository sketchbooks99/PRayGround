#include "sphere_medium.h"

namespace prayground {

SphereMedium::SphereMedium()
: m_center(make_float3(0.0f)), m_radius(1.0f), m_density(0.001f)
{

}

SphereMedium::SphereMedium(const float3& center, const float radius, const float density)
: m_center(center), m_radius(radius), m_density(density)
{

}

constexpr ShapeType SphereMedium::type()
{
    return ShapeType::Custom;
}

void SphereMedium::copyToDevice() 
{
    SphereMediumData data = this->deviceData();
    if (!d_data)
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(SphereMediumData)));
    CUDA_CHECK(cudaMemcpy(
        d_data, &data, sizeof(SphereMediumData),
        cudaMemcpyHostToDevice
    ));
}

void SphereMedium::free() 
{
    Shape::free();
    cuda_free(d_aabb_buffer);
}

OptixBuildInput SphereMedium::createBuildInput() 
{
    if (d_aabb_buffer) cuda_free(d_aabb_buffer);
    return createSingleCustomBuildInput(d_aabb_buffer, this->bound(), m_sbt_index);
}

AABB SphereMedium::bound() const 
{
    return AABB(m_center - m_radius, m_center + m_radius);
}

const float3& SphereMedium::center() const
{
    return m_center;
}

const float& SphereMedium::radius() const
{
    return m_radius;
}

SphereMedium::DataType SphereMedium::deviceData() const
{
    SphereMediumData data = 
    {
        .center = m_center,
        .radius = m_radius,
        .density = m_density
    };
    return data;
}

} // ::prayground