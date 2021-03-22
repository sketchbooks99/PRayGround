#pragma once 

#include <include/core/shape.h>
#include "optix/sphere.cuh"

namespace pt {

class Sphere : public Shape {
public:
    explicit Sphere(float3 c, float r) : m_center(c), m_radius(r) {}

    ShapeType type() const override { return ShapeType::Sphere; }

    void prepare_data() override 
    {
        SphereData data = {
            m_center, 
            m_radius, 
        };

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data_ptr), sizeof(SphereData)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_data_ptr),
            &data, sizeof(SphereData),
            cudaMemcpyHostToDevice
        ));

        Message("Sphere::prepare_data():", "d_data_ptr =", d_data_ptr);
    }

    void build_input( OptixBuildInput& bi, uint32_t sbt_idx ) override
    {
        // Prepare bounding box information on the device.
        OptixAabb aabb = (OptixAabb)this->bound();
        if (d_aabb_buffer) free_aabb_buffer();

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