#include "cylinder.h"
#include <prayground/core/cudabuffer.h>

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
OptixBuildInputType Cylinder::buildInputType() const 
{
    return OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
}

// ------------------------------------------------------------------
void Cylinder::copyToDevice()
{
    CylinderData data = {
        .radius = m_radius,
        .height = m_height
    };

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
    OptixBuildInput bi;
    CUDABuffer<uint32_t> d_sbt_indices;
    uint32_t* sbt_indices = new uint32_t[1];
    sbt_indices[0] = m_sbt_index;
    d_sbt_indices.copyToDevice(sbt_indices, sizeof(uint32_t));

    OptixAabb aabb = static_cast<OptixAabb>(bound());

    if (d_aabb_buffer) free();

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&d_aabb_buffer), sizeof(OptixAabb) ) );
    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>(d_aabb_buffer), 
        &aabb, 
        sizeof(OptixAabb), 
        cudaMemcpyHostToDevice
    ));

    unsigned int* input_flags = new unsigned int[1];
    input_flags[0] = OPTIX_GEOMETRY_FLAG_NONE;

    bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
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

} // ::prayground