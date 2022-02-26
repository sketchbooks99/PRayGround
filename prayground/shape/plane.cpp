#include "plane.h"
#include <prayground/core/util.h>

namespace prayground {

// ------------------------------------------------------------------
Plane::Plane()
: m_min{-1.0f, -1.0f}, m_max{1.0f, 1.0f}
{

}

Plane::Plane(const float2& min, const float2& max)
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

// ------------------------------------------------------------------
void Plane::free()
{
    Shape::free();
    cuda_free(d_aabb_buffer);
}

// ------------------------------------------------------------------
AABB Plane::bound() const 
{
    AABB box{make_float3(m_min.x, -0.01f, m_min.y), make_float3(m_max.x, 0.01f, m_max.y)};
    return box;
}

// ------------------------------------------------------------------
Plane::Data Plane::getData() const 
{
    return { m_min, m_max };
}

} // ::prayground