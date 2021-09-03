#pragma once
#include <optix.h>
#include <prayground/core/shape.h>
#include "context.h"
#include "instance.h"

namespace prayground {

class GeometryAccel {
public:
    GeometryAccel() = default;
    GeometryAccel(ShapeType shape_type);
    ~GeometryAccel();

    void addShape(const std::shared_ptr<Shape>& shape);
   
    void build(const Context& ctx, CUstream stream);
    void update(const Context& ctx, CUstream stream);
    void free();

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
    CUdeviceptr deviceBuffer() const;
    size_t deviceBufferSize() const;
private:
    ShapeType m_shape_type;
    OptixTraversableHandle m_handle{ 0 };
    OptixAccelBuildOptions m_options{};
    uint32_t m_count{ 0 };

    std::vector<std::shared_ptr<Shape>> m_shapes;
    std::vector<OptixBuildInput> m_build_inputs;

    bool is_hold_temp_buffer{ false };
    CUdeviceptr d_buffer{ 0 };
    size_t d_buffer_size{ 0 };
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

    void build(const Context& ctx, CUstream stream);
    void update(const Context& ctx, CUstream stream);
    void free();

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
    CUdeviceptr deviceBuffer() const;
    size_t deviceBufferSize() const;
private:
    Type m_type;
    OptixTraversableHandle m_handle{ 0 };
    OptixAccelBuildOptions m_options{};
    uint32_t m_count{ 0 };

    std::vector<Instance> m_instances;
    CUdeviceptr d_instances;
    OptixBuildInput m_instance_input;

    CUdeviceptr d_buffer{ 0 };
    size_t d_buffer_size{ 0 };
};

}