#pragma once

#include <variant>
#include <memory>
#include <prayground/core/util.h>
#include <prayground/core/aabb.h>
#include <prayground/core/material.h>
#include <prayground/core/interaction.h>
#include <prayground/emitter/area.h>
#include <prayground/optix/macros.h>
#include <prayground/optix/program.h>
#include <sutil/vec_math.h>
#include <optix_types.h>

namespace prayground {

#ifndef __CUDACC__

class Shape {
public:
    virtual ~Shape() {}

    virtual OptixBuildInputType buildInputType() const = 0;
    virtual AABB bound() const = 0;

    virtual void copyToDevice() = 0;
    virtual void buildInput( OptixBuildInput& bi ) = 0;

    void attachSurface(const std::shared_ptr<Material>& material);
    void attachSurface(const std::shared_ptr<AreaEmitter>& area_emitter);
    std::variant<std::shared_ptr<Material>, std::shared_ptr<AreaEmitter>> surface() const;
    SurfaceType surfaceType() const;
    void* surfaceDevicePtr() const;

    void setSbtIndex(const uint32_t sbt_index);
    uint32_t sbtIndex() const;

    void free();
    void freeAabbBuffer();
    
    void* devicePtr() const;

protected:
    void* d_data { 0 };
    CUdeviceptr d_aabb_buffer { 0 };
    uint32_t m_sbt_index;

private:
    std::variant<std::shared_ptr<Material>, std::shared_ptr<AreaEmitter>> m_surface;
};

#endif // __CUDACC__

}