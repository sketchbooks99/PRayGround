#pragma once

#include <optix_types.h>
#include <prayground/core/aabb.h>
#include <prayground/optix/macros.h>

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

    virtual ShapeType type() const = 0;

    virtual void copyToDevice() = 0;
    virtual OptixBuildInput createBuildInput() = 0;

    void setSbtIndex(const uint32_t sbt_index);
    uint32_t sbtIndex() const;

    virtual AABB bound() const = 0;

    virtual void free();

    void* devicePtr() const;

protected:
    void* d_data { 0 };
    uint32_t m_sbt_index;
};

#endif // __CUDACC__

}