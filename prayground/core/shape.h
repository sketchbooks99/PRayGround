#pragma once

#include <memory>
#include <optix_types.h>
#include <cuda_runtime.h>
#include <prayground/core/util.h>
#include <prayground/core/aabb.h>
#include <prayground/optix/macros.h>

namespace prayground {

#ifndef __CUDACC__

class Shape {
public:
    virtual ~Shape() {}

    virtual OptixBuildInputType buildInputType() const = 0;

    virtual void copyToDevice() = 0;
    virtual OptixBuildInput createBuildInput() = 0;

    void setSbtIndex(const uint32_t sbt_index);
    uint32_t sbtIndex() const;

    virtual void free();

    void* devicePtr() const;

protected:
    void* d_data { 0 };
    uint32_t m_sbt_index;
};

#endif // __CUDACC__

}