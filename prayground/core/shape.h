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

#endif // __CUDACC__

}