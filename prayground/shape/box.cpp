#include "box.h"
#include <prayground/core/cudabuffer.h>
#include <prayground/core/util.h>

namespace prayground {

// ------------------------------------------------------------------
Box::Box()
: m_min(make_float3(-1.0f)), m_max(make_float3(1.0f))
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
    BoxData data = this->deviceData();

    if (!d_data)
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(BoxData)));
    CUDA_CHECK(cudaMemcpy(
        d_data, 
        &data, sizeof(BoxData), 
        cudaMemcpyHostToDevice
    ));
}

// ------------------------------------------------------------------
OptixBuildInput Box::createBuildInput()
{
    OptixBuildInput bi = {};
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

// ------------------------------------------------------------------
Box::DataType Box::deviceData() const 
{
    BoxData data = 
    {
        .min = m_min,
        .max = m_max
    };

    return data;
}

} // ::prayground