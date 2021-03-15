#pragma once 

#include <core/shape.h>
#include "optix/sphere.cuh"

namespace pt {

class Sphere : public Shape {
public:
    explicit Sphere(float3 c, float r) : m_center(c), m_radius(r) {}

    HOST ShapeType type() const override { return ShapeType::Sphere; }
    HOST AABB bound() const override { 
        return AABB( m_center - make_float3(m_radius),
                     m_center + make_float3(m_radius) );
    }
    HOST void build_input( OptixBuildInput& bi, uint32_t sbt_idx ) const override
    {
        // Prepare bounding box information on the device.
        OptixAabb aabb = (OptixAabb)this->bound();
        CUdeviceptr d_aabb_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), sizeof(OptixAabb)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_aabb_buffer),
            &aabb,
            sizeof(OptixAabb),
            cudaMemcpyHostToDevice));

        bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        bi.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
        bi.customPrimitiveArray.flags = (unsigned int*)(OPTIX_GEOMETRY_FLAG_NONE);
        bi.customPrimitiveArray.numSbtRecords = 1;
    }
private:
    float3 m_center;
    float m_radius;
};

}