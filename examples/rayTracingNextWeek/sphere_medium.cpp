#include "sphere_medium.h"

namespace prayground {

SphereMedium::SphereMedium()
: m_center(Vec3f(0.0f)), m_radius(1.0f), m_density(0.001f)
{

}

SphereMedium::SphereMedium(const Vec3f& center, const float radius, const float density)
: m_center(center), m_radius(radius), m_density(density)
{

}

constexpr ShapeType SphereMedium::type()
{
    return ShapeType::Custom;
}

void SphereMedium::copyToDevice() 
{
    auto data = this->getData();
    if (!d_data)
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
    CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(Data), cudaMemcpyHostToDevice));
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

uint32_t SphereMedium::numPrimitives() const
{
    return 1;
}

AABB SphereMedium::bound() const 
{
    return AABB(m_center - m_radius, m_center + m_radius);
}

const Vec3f& SphereMedium::center() const
{
    return m_center;
}

const float& SphereMedium::radius() const
{
    return m_radius;
}

SphereMedium::Data SphereMedium::getData() const
{
    return { m_center, m_radius, m_density };
}

} // namespace prayground