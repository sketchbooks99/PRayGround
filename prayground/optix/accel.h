#pragma once
#include <optix.h>
#include <prayground/core/shape.h>
#include "context.h"
#include "instance.h"

namespace prayground {

class GeometryAccel {
public:
    enum class Type {
        Mesh = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
        Custom = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
        Curves = OPTIX_BUILD_INPUT_TYPE_CURVES,
        None
    };

    GeometryAccel() = default;
    explicit GeometryAccel(Type type);
    ~GeometryAccel();

    void addShape(const std::shared_ptr<Shape>& shape);
    std::shared_ptr<Shape> shapeAt(const int idx) const;
   
    void build(const Context& ctx);
    void update(const Context& ctx);
    void update(const Context& ctx, CUdeviceptr temp_buffer, size_t temp_buffer_size);
    void free();

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
    Type m_type;
    OptixTraversableHandle m_handle{ 0 };
    OptixAccelBuildOptions m_options{};
    uint32_t m_count{ 0 };

    std::vector<std::shared_ptr<Shape>> m_shapes;
    std::vector<OptixBuildInput> m_build_inputs;

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

    InstanceAccel() = default;
    explicit InstanceAccel(Type type);
    ~InstanceAccel();

    void addInstance(const Instance& instance);
    Instance& instanceAt(const int32_t idx);

    void build(const Context& ctx);
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

    std::vector<Instance> m_instances;
    /// @note 
    /// update() を考えると、自分自身でOptixBuildInput を保持している方がいい気がする。
    /// OptixBuildInput m_instance_input;

    bool is_hold_temp_buffer{ false };
    CUdeviceptr d_buffer{ 0 }, d_temp_buffer{ 0 };
    size_t d_buffer_size{ 0 }, d_temp_buffer_size{ 0 };
};

}