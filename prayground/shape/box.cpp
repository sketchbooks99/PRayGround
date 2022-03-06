#include "box.h"
#include <prayground/core/cudabuffer.h>
#include <prayground/core/util.h>
#include <prayground/math/util.h>

namespace prayground {

// ------------------------------------------------------------------
Box::Box()
: m_min(Vec3f(-1.0f)), m_max(Vec3f(1.0f))
{

}

Box::Box(const float3& min, const float3& max)
: m_min(min), m_max(max)
{

}

// ------------------------------------------------------------------
constexpr ShapeType Box::type()
{
    return ShapeType::Custom;
}

// ------------------------------------------------------------------
void Box::copyToDevice()
{
    Data data = this->getData();
    if (!d_data)
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
    CUDA_CHECK(cudaMemcpy(
        d_data, &data, sizeof(Data), cudaMemcpyHostToDevice
    ));
}

// ------------------------------------------------------------------
OptixBuildInput Box::createBuildInput()
{
    if (d_aabb_buffer) cuda_free(d_aabb_buffer);
    return createSingleCustomBuildInput(d_aabb_buffer, this->bound(), m_sbt_index);
}

// ------------------------------------------------------------------
void Box::free()
{
    Shape::free();
    cuda_free(d_aabb_buffer);
}

// ------------------------------------------------------------------
AABB Box::bound() const 
{
    return AABB(m_min, m_max);
}

const float3& Box::min() const
{
    return m_min;
}
const float3& Box::max() const
{
    return m_max;
}

// ------------------------------------------------------------------
Box::Data Box::getData() const 
{
    return { m_min, m_max };
}

} // ::prayground