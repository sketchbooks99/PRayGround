#pragma once
#include <optix.h>
#include "../core/shape.h"
#include "context.h"
#include "instance.h"

namespace oprt {

class GeometryAccel {
public:
    GeometryAccel();
    GeometryAccel(const OptixAccelBuildOptions& options);
    ~GeometryAccel();

    void build(const Context& ctx, const std::shared_ptr<Shape>& shape);
    void build(const Context& ctx, const std::vector<std::shared_ptr<Shape>>& shape);
    void free();

    uint32_t count() const;
    OptixTraversableHandle handle() const;
private:
    OptixTraversableHandle m_handle;
    OptixAccelBuildOptions m_options;
    uint32_t m_count;
    CUdeviceptr d_buffer;
};

class InstanceAccel {
public:
    InstanceAccel();
    InstanceAccel(const OptixAccelBuildOptions& options);

    void build(const Context& ctx, const Instance& instance);
    void build(const Context& ctx, const std::vector<Instance>& instances);
    void free();

    uint32_t count() const;
    OptixTraversableHandle handle() const;
private:
    OptixTraversableHandle m_handle;
    OptixAccelBuildOptions m_options;
    uint32_t m_count;
    CUdeviceptr d_buffer;
};

}