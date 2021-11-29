#include "shape.h"
#include <prayground/core/util.h>

namespace prayground {

void Shape::setSbtIndex(const uint32_t sbt_idx)
{
    m_sbt_index = sbt_idx;
}

uint32_t Shape::sbtIndex() const
{
    return m_sbt_index;
}

void Shape::free()
{
    if (d_data) CUDA_CHECK(cudaFree(d_data));
    d_data = nullptr;
}

void* Shape::devicePtr() const
{
    return d_data;
}

OptixBuildInput createSingleCustomBuildInput(
    CUdeviceptr& d_aabb_buffer,
    AABB bound, 
    uint32_t sbt_index, 
    uint32_t flags
)
{
    OptixBuildInput bi = {};
    CUDABuffer<uint32_t> d_sbt_indices;
    uint32_t* sbt_indices = new uint32_t[1];
    sbt_indices[0] = sbt_index;
    d_sbt_indices.copyToDevice(sbt_indices, sizeof(uint32_t));

    OptixAabb aabb = static_cast<OptixAabb>(bound);

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

} // ::prayground
