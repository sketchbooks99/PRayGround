#include "sphere.h"
#include <prayground/core/cudabuffer.h>
#include <prayground/core/util.h>

namespace prayground {

// ------------------------------------------------------------------
Sphere::Sphere()
: m_center{0.0f, 0.0f, 0.0f}, m_radius{1.0f}
{

}

Sphere::Sphere(const float3& c, float r)
: m_center(c), m_radius(r)
{

}

// ------------------------------------------------------------------
constexpr ShapeType Sphere::type()
{
    return ShapeType::Custom;
}

// ------------------------------------------------------------------
void Sphere::copyToDevice()
{
    SphereData data = this->deviceData();

    if (!d_data)
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(SphereData)));
    CUDA_CHECK(cudaMemcpy(
        d_data, 
        &data, sizeof(SphereData), 
        cudaMemcpyHostToDevice
    ));
}

// ------------------------------------------------------------------
OptixBuildInput Sphere::createBuildInput()
{
    if (d_aabb_buffer) cuda_free(d_aabb_buffer);
    return createSingleCustomBuildInput(d_aabb_buffer, this->bound(), m_sbt_index);
} 

// ------------------------------------------------------------------
AABB Sphere::bound() const 
{ 
    return AABB( m_center - make_float3(m_radius),
                 m_center + make_float3(m_radius) );
}

// ------------------------------------------------------------------
Sphere::DataType Sphere::deviceData() const 
{
    SphereData data = 
    {
        .center = m_center, 
        .radius = m_radius
    };
    return data;
}

} // ::prayground