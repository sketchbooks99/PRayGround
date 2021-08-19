#pragma once
#include <optix.h>
#include <oprt/core/shape.h>
#include "context.h"
#include "instance.h"

namespace oprt {

class GeometryAccel {
public:
    GeometryAccel();
    ~GeometryAccel();

    void build(const Context& ctx, const std::shared_ptr<Shape>& shape);
    void build(const Context& ctx, const std::vector<std::shared_ptr<Shape>>& shapes);
    void update(const Context& ctx);
    void free();

    /**  */
    void enableHoldTempBuffer();
    void disableHoldTempBuffer();

    void setFlags(const uint32_t build_flags);

    void allowUpdate();
    void allowCompaction();
    void preferFastTrace();
    void preferFastBuild();
    void allowRandomVertexAccess();

    void disableUpdate();
    void disableCompaction();
    void disableFastTrace();
    void disableFastBuild();
    void disableRandomVertexAccess();

    void setMotionOptions(const OptixMotionOptions& motion_options);

    uint32_t count() const;
    OptixTraversableHandle handle() const;
    bool isTempBuffer() const;
    CUdeviceptr deviceBuffer() const;
    size_t deviceBufferSize() const;
    CUdeviceptr deviceTempBuffer() const;
    size_t deviceTempBufferSize() const;
private:
    OptixTraversableHandle m_handle{ 0 };
    OptixAccelBuildOptions m_options{};
    uint32_t m_count{ 0 };

    bool is_hold_temp_buffer{ false };
    CUdeviceptr d_buffer{ 0 }, d_temp_buffer{ 0 };
    size_t d_buffer_size{ 0 }, d_temp_buffer_size{ 0 };
};

class InstanceAccel {
public:
    enum class Type
    {
        Instances = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
        InstancePointers = OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS
    };

    InstanceAccel();
    InstanceAccel(Type type);
    ~InstanceAccel();

    void build(const Context& ctx, const std::shared_ptr<Instance>& instance);
    void build(const Context& ctx, const OptixInstance& optix_instance);
    void build(const Context& ctx, const std::vector<std::shared_ptr<Instance>>& instances);
    void build(const Context& ctx, const std::vector<OptixInstance>& optix_instances);
    void update(const Context& ctx);
    void free();

    /** Switch flag whether to enable store  */
    void enableHoldTempBuffer();
    void disableHoldTempBuffer();

    void setFlags(const uint32_t build_flags);
    void allowUpdate();
    void allowCompaction();
    void preferFastTrace();
    void preferFastBuild();

    void disableUpdate();
    void disableCompaction();
    void disableFastTrace();
    void disableFastBuild();

    void setMotionOptions(const OptixMotionOptions& motion_options);

    uint32_t count() const;
    OptixTraversableHandle handle() const;
    bool isTempBuffer() const;
    CUdeviceptr deviceBuffer() const;
    size_t deviceBufferSize() const;
    CUdeviceptr deviceTempBuffer() const;
    size_t deviceTempBufferSize() const;
private:
    Type m_type;
    OptixTraversableHandle m_handle{ 0 };
    OptixAccelBuildOptions m_options{};
    uint32_t m_count{ 0 };

    bool is_hold_temp_buffer{ false };
    CUdeviceptr d_buffer{ 0 }, d_temp_buffer{ 0 };
    size_t d_buffer_size{ 0 }, d_temp_buffer_size{ 0 };
};

}