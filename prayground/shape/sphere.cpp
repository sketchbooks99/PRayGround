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
ShapeType Sphere::type() const
{
    return ShapeType::Custom;
}

// ------------------------------------------------------------------
void Sphere::copyToDevice()
{
    SphereData data = {
        .center = m_center, 
        .radius = m_radius
    };

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
    OptixBuildInput bi = {};
    CUDABuffer<uint32_t> d_sbt_indices;
    uint32_t* sbt_indices = new uint32_t[1];
    sbt_indices[0] = m_sbt_index;
    d_sbt_indices.copyToDevice(sbt_indices, sizeof(uint32_t));

    // Prepare bounding box information on the device.
    OptixAabb aabb = static_cast<OptixAabb>(bound());

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_aabb_buffer),
        &aabb,
        sizeof(OptixAabb),
        cudaMemcpyHostToDevice));

    uint32_t* input_flags = new uint32_t[1];
    input_flags[0] = OPTIX_GEOMETRY_FLAG_NONE;

    bi.type = static_cast<OptixBuildInputType>(this->type());
    bi.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
    bi.customPrimitiveArray.numPrimitives = 1;
    bi.customPrimitiveArray.flags = input_flags;
    bi.customPrimitiveArray.numSbtRecords = 1;
    bi.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_indices.devicePtr();
    bi.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    bi.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
    return bi;
} 

// ------------------------------------------------------------------
AABB Sphere::bound() const 
{ 
    return AABB( m_center - make_float3(m_radius),
                 m_center + make_float3(m_radius) );
}

} // ::prayground