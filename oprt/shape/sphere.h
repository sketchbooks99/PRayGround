#pragma once 

#include "optix/sphere.cuh"

#ifndef __CUDACC__
#include "../core/shape.h"

namespace oprt {

class Sphere final : public Shape {
public:
    explicit Sphere(float3 c, float r) : m_center(c), m_radius(r) {}

    ShapeType type() const override { return ShapeType::Sphere; }

    void prepareData() override 
    {
        SphereData data = {
            m_center, 
            m_radius
        };

        CUDA_CHECK( cudaMalloc( &d_data, sizeof(SphereData) ) );
        CUDA_CHECK( cudaMemcpy(
            d_data,
            &data, sizeof(SphereData),
            cudaMemcpyHostToDevice
        ));
    }

    /**
     * @note \c index_offset is not needed.
     */
    void buildInput( OptixBuildInput& bi, uint32_t sbt_idx ) override
    {
        CUDABuffer<uint32_t> d_sbt_indices;
        uint32_t* sbt_indices = new uint32_t[1];
        sbt_indices[0] = sbt_idx;
        d_sbt_indices.copyToDevice(sbt_indices, sizeof(uint32_t));

        // Prepare bounding box information on the device.
        OptixAabb aabb = (OptixAabb)this->bound();

        if (d_aabb_buffer) freeAabbBuffer();

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), sizeof(OptixAabb)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_aabb_buffer),
            &aabb,
            sizeof(OptixAabb),
            cudaMemcpyHostToDevice));

        unsigned int* input_flags = new unsigned int[1];
        input_flags[0] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

        bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        bi.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
        bi.customPrimitiveArray.numPrimitives = 1;
        bi.customPrimitiveArray.flags = input_flags;
        bi.customPrimitiveArray.numSbtRecords = 1;
        bi.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_indices.devicePtr();
        bi.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        bi.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
    }

    AABB bound() const override { 
        return AABB( m_center - make_float3(m_radius),
                     m_center + make_float3(m_radius) );
    }
private:
    float3 m_center;
    float m_radius;
};

}

#endif