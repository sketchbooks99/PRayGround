#pragma once
#include <optix.h>
#include "../core/shape.h"
#include "context.h"
#include "instance.h"

namespace oprt {

class GeometryAccel {
public:
    GeometryAccel();
    ~GeometryAccel();

    void build(const Context& ctx, const std::shared_ptr<Shape>& shape, uint32_t sbt_base);
    void build(const Context& ctx, const std::vector<std::shared_ptr<Shape>>& shapes, uint32_t sbt_base);
    void update(const Context& ctx);
    void free();

    void setFlags(const uint32_t build_flags);
    void setMotionOptions(const OptixMotionOptions& motion_options);

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
    enum class Type
    {
        Instances = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        InstancePointers = OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS
    };

    InstanceAccel();
    InstanceAccel(InstanceAccel::Type);
    ~InstanceAccel();

    void build(const Context& ctx, const Instance& instance);
    void build(const Context& ctx, const OptixInstance& optix_instance);
    void build(const Context& ctx, const std::vector<Instance>& instances);
    void build(const Context& ctx, const std::vector<OptixInstance>& optix_instances);
    void update(const Context& ctx);
    void free();

    void setFlags(const uint32_t build_flags);
    void setMotionOptions(const OptixMotionOptions& motion_options);

    uint32_t count() const;
    OptixTraversableHandle handle() const;
private:
    OptixTraversableHandle m_handle;
    OptixAccelBuildOptions m_options;
    Type m_type;
    uint32_t m_count;
    CUdeviceptr d_buffer;
};

}