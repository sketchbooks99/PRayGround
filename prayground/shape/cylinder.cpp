#include "cylinder.h"
#include <prayground/core/cudabuffer.h>
#include <prayground/core/util.h>

namespace prayground {

// ------------------------------------------------------------------
Cylinder::Cylinder()
: m_radius(0.5f), m_height(1.0f)
{

}

Cylinder::Cylinder(float radius, float height)
: m_radius(radius), m_height(height)
{

}

// ------------------------------------------------------------------
constexpr ShapeType Cylinder::type()
{
    return ShapeType::Custom;
}

// ------------------------------------------------------------------
void Cylinder::copyToDevice()
{
    CylinderData data = this->deviceData();

    if (!d_data) 
        CUDA_CHECK( cudaMalloc( &d_data, sizeof(CylinderData) ) );
    CUDA_CHECK( cudaMemcpy(
        d_data, 
        &data, sizeof(CylinderData), 
        cudaMemcpyHostToDevice
    ));
}

// ------------------------------------------------------------------
OptixBuildInput Cylinder::createBuildInput()
{
    if (d_aabb_buffer) free();
    return createSingleCustomBuildInput(d_aabb_buffer, this->bound(), m_sbt_index); 
}

// ------------------------------------------------------------------
void Cylinder::free()
{
    Shape::free();
    cuda_free(d_aabb_buffer);
}

// ------------------------------------------------------------------
AABB Cylinder::bound() const 
{
    return AABB( 
        -make_float3(m_radius, m_height / 2.0f, m_radius),
         make_float3(m_radius, m_height / 2.0f, m_radius)
    );
}

// ------------------------------------------------------------------
Cylinder::DataType Cylinder::deviceData() const 
{
    CylinderData data = 
    {
        .radius = m_radius,
        .height = m_height
    };

    return data;
}

} // ::prayground