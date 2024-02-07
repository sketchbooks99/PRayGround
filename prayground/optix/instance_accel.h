#pragma once 
#include <optix.h>
#include <prayground/optix/context.h>
#include <prayground/optix/instance.h>
#include <variant>

namespace prayground {

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
        void addInstance(const ShapeInstance& shape_instance);

        void build(const Context& ctx, CUstream stream);
        void update(const Context& ctx, CUstream stream);
        // void relocate(); // TODO
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
        const OptixMotionOptions& motionOptions() const;

        uint32_t count() const;
        OptixTraversableHandle handle() const;
        CUdeviceptr deviceBuffer() const;
        size_t deviceBufferSize() const;
        bool isBuilded() const;
    private:
        Type m_type;
        OptixTraversableHandle m_handle{ 0 };
        OptixAccelBuildOptions m_options{};
        uint32_t m_count{ 0 };

        std::vector<OptixInstance*> m_instances;
        CUdeviceptr d_instances;
        OptixBuildInput m_instance_input;

        CUdeviceptr d_buffer{ 0 };
        size_t d_buffer_size{ 0 };
    };

} // namespace prayground