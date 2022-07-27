#pragma once

#include <optix_types.h>
#include <prayground/core/aabb.h>
#include <prayground/optix/macros.h>

#ifndef __CUDACC__
#include <prayground/core/cudabuffer.h>
#endif

namespace prayground {

    enum class ShapeType
    {
        Mesh = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
        Custom = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
        Curves = OPTIX_BUILD_INPUT_TYPE_CURVES
    };

    class Shape {

    /// @note Make this class be dummy class on device kernels
#ifndef __CUDACC__

    public:
        virtual ~Shape() {}

        virtual constexpr ShapeType type() = 0;

        virtual void copyToDevice() = 0;
        virtual AABB bound() const = 0;

        virtual OptixBuildInput createBuildInput() = 0;

        virtual uint32_t numPrimitives() const = 0;

        virtual void free();

        virtual void setSbtIndex(const uint32_t sbt_index);
        virtual uint32_t sbtIndex() const;

        void* devicePtr() const;

    protected:
        void* d_data { nullptr };
        uint32_t m_sbt_index { 0 };

#endif // __CUDACC__

    };

#ifndef __CUDACC__

    OptixBuildInput createSingleCustomBuildInput(
        CUdeviceptr& d_aabb_buffer,
        AABB bound, 
        uint32_t sbt_index, 
        uint32_t flags = OPTIX_GEOMETRY_FLAG_NONE
    );

#endif // __CUDACC__


}