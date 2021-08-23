#pragma once 

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#endif

namespace prayground {

struct SphereData {
    float3 center;
    float radius;
};

#ifndef __CUDACC__
class Sphere final : public Shape {
public:
    explicit Sphere(const float3& c, float r) : m_center(c), m_radius(r) {}

    OptixBuildInputType buildInputType() const override { return OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; }

    void copyToDevice() override 
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

    void buildInput( OptixBuildInput& bi ) override
    {
        CUDABuffer<uint32_t> d_sbt_indices;
        uint32_t* sbt_indices = new uint32_t[1];
        sbt_indices[0] = m_sbt_index;
        d_sbt_indices.copyToDevice(sbt_indices, sizeof(uint32_t));

        // Prepare bounding box information on the device.
        OptixAabb aabb = static_cast<OptixAabb>(bound());

        if (d_aabb_buffer) free();

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

#endif

}