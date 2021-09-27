#pragma once

#include <optix_types.h>
#include <prayground/core/aabb.h>
#include <prayground/optix/macros.h>

#ifndef __CUDACC__
#include <prayground/core/cudabuffer.h>
#endif

namespace prayground {

#ifndef __CUDACC__

enum class ShapeType
{
    Mesh = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
    Custom = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
    Curves = OPTIX_BUILD_INPUT_TYPE_CURVES
};

class Shape {
public:
    virtual ~Shape() {}

    virtual constexpr ShapeType type() = 0;

    virtual void copyToDevice() = 0;
    virtual AABB bound() const = 0;

    virtual OptixBuildInput createBuildInput() = 0;

    virtual void free();

    void setSbtIndex(const uint32_t sbt_index);
    uint32_t sbtIndex() const;

    void* devicePtr() const;

protected:
    void* d_data { nullptr };
    uint32_t m_sbt_index { 0 };
};

inline OptixBuildInput createSingleCustomBuildInput(
    CUdeviceptr& d_aabb_buffer,
    AABB bound, 
    uint32_t sbt_index, 
    uint32_t flags = OPTIX_GEOMETRY_FLAG_NONE)
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

#endif // __CUDACC__

}