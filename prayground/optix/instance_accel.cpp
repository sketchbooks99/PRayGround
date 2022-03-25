#include "instance_accel.h"
#include <prayground/optix/macros.h>
#include <prayground/math/util.h>
#include <algorithm>

namespace prayground {

InstanceAccel::InstanceAccel(Type type)
: m_type(type)
{
    m_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
}

InstanceAccel::~InstanceAccel()
{

}

// ---------------------------------------------------------------------------
void InstanceAccel::addInstance(const Instance& instance)
{
    m_instances.emplace_back(instance.rawInstancePtr());
}

void InstanceAccel::addInstance(const ShapeInstance& shape_instance)
{
    m_instances.emplace_back(shape_instance.rawInstancePtr());
}

// ---------------------------------------------------------------------------
void InstanceAccel::build(const Context& ctx, CUstream stream)
{
    std::vector<OptixInstance> optix_instances;
    std::transform(m_instances.begin(), m_instances.end(), std::back_inserter(optix_instances),
        [](OptixInstance* instance) { return *instance; });

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), sizeof(OptixInstance) * optix_instances.size()));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_instances),
        optix_instances.data(), sizeof(OptixInstance) * optix_instances.size(),
        cudaMemcpyHostToDevice
    ));

    m_instance_input = {};
    m_instance_input.type = static_cast<OptixBuildInputType>(m_type);
    m_instance_input.instanceArray.instances = d_instances;
    m_instance_input.instanceArray.numInstances = static_cast<uint32_t>(m_instances.size());

    m_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        static_cast<OptixDeviceContext>(ctx), 
        &m_options, 
        &m_instance_input, 
        1,  // num build inputs
        &ias_buffer_sizes ));
    d_buffer_size = ias_buffer_sizes.outputSizeInBytes;
    size_t d_temp_buffer_size = ias_buffer_sizes.tempSizeInBytes;
        
    // Allocate buffer to build acceleration structure
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer), 
        d_temp_buffer_size ));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer), 
        d_buffer_size ));
    
    // Build instance AS contains all GASs to describe the scene.
    OPTIX_CHECK(optixAccelBuild(
        static_cast<OptixDeviceContext>(ctx), 
        stream,                  // CUDA stream
        &m_options, 
        &m_instance_input, 
        1,                  // num build inputs
        d_temp_buffer, 
        d_temp_buffer_size, 
        d_buffer, 
        d_buffer_size, 
        &m_handle, 
        nullptr,            // emitted property list
        0                   // num emitted properties
    ));

    cuda_free(d_temp_buffer);
}

void InstanceAccel::update(const Context& ctx, CUstream stream)
{
    ASSERT((m_options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0, "allowUpdate() must be called before an update operation.");

    OptixInstance* instance_device_ptr = reinterpret_cast<OptixInstance*>(d_instances);
    for (int i=0; OptixInstance* instance : m_instances)
    {
        CUDA_CHECK(cudaMemcpy(
            &instance_device_ptr[i],
            instance, 
            sizeof(OptixInstance),
            cudaMemcpyHostToDevice
        ));
        i++;
    }

    m_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        static_cast<OptixDeviceContext>(ctx), 
        &m_options, 
        &m_instance_input,
        1,
        &gas_buffer_sizes
    ));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempUpdateSizeInBytes));
    size_t d_temp_buffer_size = gas_buffer_sizes.tempUpdateSizeInBytes;

    OPTIX_CHECK(optixAccelBuild(
        static_cast<OptixDeviceContext>(ctx),
        stream,
        &m_options,
        &m_instance_input,
        1,
        d_temp_buffer,
        d_temp_buffer_size,
        d_buffer,
        d_buffer_size,
        &m_handle,
        nullptr,
        0
    ));

    CUDA_SYNC_CHECK();
}

void InstanceAccel::free()
{
    if (d_buffer) cuda_free(d_buffer);
    d_buffer_size = 0;
}

// ---------------------------------------------------------------------------
void InstanceAccel::setFlags(const uint32_t build_flags)
{
    m_options.buildFlags = build_flags;
}

void InstanceAccel::allowUpdate()
{
    m_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
}

void InstanceAccel::allowCompaction()
{
    m_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
}

void InstanceAccel::preferFastTrace()
{
    m_options.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
}

void InstanceAccel::preferFastBuild()
{
    m_options.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
}

void InstanceAccel::disableUpdate()
{
    m_options.buildFlags &= ~OPTIX_BUILD_FLAG_ALLOW_UPDATE;
}

void InstanceAccel::disableCompaction()
{
    m_options.buildFlags &= ~OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
}

void InstanceAccel::disableFastTrace()
{
    m_options.buildFlags &= ~OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
}

void InstanceAccel::disableFastBuild()
{
    m_options.buildFlags &= ~OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
}

void InstanceAccel::setMotionOptions(const OptixMotionOptions& motion_options)
{
    m_options.motionOptions = motion_options;
}

const OptixMotionOptions& InstanceAccel::motionOptions() const
{
    return m_options.motionOptions;
}

// ---------------------------------------------------------------------------
uint32_t InstanceAccel::count() const
{
    return m_count;
}

OptixTraversableHandle InstanceAccel::handle() const
{
    return m_handle;
}

CUdeviceptr InstanceAccel::deviceBuffer() const
{
    return d_buffer;
}

size_t InstanceAccel::deviceBufferSize() const
{
    return d_buffer_size;
}

bool InstanceAccel::isBuilded() const
{
    return (bool)d_buffer;
}

} // ::prayground