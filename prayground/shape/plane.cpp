#include "plane.h"

namespace prayground {

// ------------------------------------------------------------------
Plane::Plane()
: m_min{-1.0f, -1.0f}, m_max{1.0f, 1.0f}
{

}

Plane::Plane(const float2& min, const float2& max)
: m_min{ min }, m_max{ max }
{

}

// ------------------------------------------------------------------
ShapeType Plane::type() const
{
    return ShapeType::Custom;
}

// ------------------------------------------------------------------
void Plane::copyToDevice() 
{
    PlaneData data = {
        .min = m_min, 
        .max = m_max
    };

    if (!d_data)
        CUDA_CHECK( cudaMalloc( &d_data, sizeof(PlaneData) ) );
    CUDA_CHECK( cudaMemcpy(
        d_data, 
        &data, sizeof(PlaneData), 
        cudaMemcpyHostToDevice
    ));
}

// ------------------------------------------------------------------
OptixBuildInput Plane::createBuildInput()
{
    OptixBuildInput bi = {};
    CUDABuffer<uint32_t> d_sbt_indices;
    uint32_t* sbt_indices = new uint32_t[1];
    sbt_indices[0] = m_sbt_index;
    d_sbt_indices.copyToDevice(sbt_indices, sizeof(uint32_t));

    // Prepare bounding box information on the device.
    OptixAabb aabb = static_cast<OptixAabb>(this->bound());

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_aabb_buffer),
        &aabb,
        sizeof(OptixAabb),
        cudaMemcpyHostToDevice));

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
void Plane::free()
{
    Shape::free();
    cuda_free(d_aabb_buffer);
}

// ------------------------------------------------------------------
AABB Plane::bound() const 
{
    AABB box{make_float3(m_min.x, -0.01f, m_min.y), make_float3(m_max.x, 0.01f, m_max.y)};
    return box;
}

} // ::prayground