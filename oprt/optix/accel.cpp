#include "accel.h"
#include "../core/cudabuffer.h"

namespace oprt {

// GeometryAccel -------------------------------------------------------------
GeometryAccel::GeometryAccel()
{
    m_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
}

GeometryAccel::~GeometryAccel()
{

}

// ---------------------------------------------------------------------------
void GeometryAccel::build(const Context& ctx, const std::shared_ptr<Shape>& shape, uint32_t sbt_base)
{
    std::vector<std::shared_ptr<Shape>> shapes(1, shape);
    build(ctx, shapes, sbt_base);
}

void GeometryAccel::build(const Context& ctx, const std::vector<std::shared_ptr<Shape>>& shapes, uint32_t sbt_base)
{
    if (shapes.size() == 0)
    {
        Message(MSG_ERROR, "oprt::GeoetryAccel::build(): The number of shapes is 0.");
        return;
    }

    bool is_all_same_type = true;
    OptixBuildInputType zeroth_input_type = shapes[0]->buildInputType();

    if (zeroth_input_type == OPTIX_BUILD_INPUT_TYPE_INSTANCES || OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS)
    {
        Message(MSG_ERROR, "oprt::GeometryAccel::build(): The OptixBuildInputType of","OPTIX_BUILD_INPUT_TYPE_INSTANCES or OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS cannot be used as an input type of geometry acceleration structure.");
        return;
    }

    for (auto& shape : shapes)
        is_all_same_type &= (zeroth_input_type == shape->buildInputType());
    if (!is_all_same_type)
    {
        Message(MSG_ERROR, "oprt::GeometryAccel::build(): All type of shapes must be a same type.");
        return;
    }

    if (d_buffer)
    {
        cuda_free(d_buffer);
        m_handle = 0;
        d_buffer = 0; 
        m_count = 0;
    }

    std::vector<OptixBuildInput> build_inputs(shapes.size());
    m_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    uint32_t sbt_idx = sbt_base;
    for (size_t i = 0; i < shapes.size(); i++)
    {
        shapes[i]->buildInput(build_inputs[i], sbt_idx);
        sbt_idx++;
    }

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        static_cast<OptixDeviceContext>(ctx), 
        &m_options, 
        build_inputs.data(), 
        static_cast<uint32_t>(build_inputs.size()), 
        &gas_buffer_sizes
    ));

    // Temporarily buffer to build GAS
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compacted_size_offset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), 
        compacted_size_offset + 8
    ));

    OptixAccelEmitDesc emit_property = {};
    emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compacted_size_offset);

    OPTIX_CHECK(optixAccelBuild(
        static_cast<OptixDeviceContext>(ctx),
        0, 
        &m_options, 
        build_inputs.data(), 
        build_inputs.size(), 
        d_temp_buffer, 
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes, 
        &m_handle, 
        &emit_property, 
        1
    ));

    // Free temporarily buffers 
    cuda_free(d_temp_buffer);

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emit_property.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer), compacted_gas_size));
        OPTIX_CHECK(optixAccelCompact(static_cast<OptixDeviceContext>(ctx), 0, m_handle, d_buffer, compacted_gas_size, &m_handle));
        cuda_free(d_buffer_temp_output_gas_and_compacted_size);
    }
    else {
        d_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void GeometryAccel::update(const Context& ctx)
{
    /** Future work */
}

void GeometryAccel::free()
{
    if (d_buffer) cudaFree(reinterpret_cast<void*>(d_buffer));
}

// -----------------------------------C----------------------------------------
void GeometryAccel::setFlags(const uint32_t build_flags)
{
    m_options.buildFlags = build_flags;
}

void GeometryAccel::setMotionOptions(const OptixMotionOptions& motion_options)
{
    m_options.motionOptions = motion_options;
}

// -----------------------------------C----------------------------------------
uint32_t GeometryAccel::count() const
{
    return m_count;
}

OptixTraversableHandle GeometryAccel::handle() const
{
    return m_handle;
}

// InstanceAccel -------------------------------------------------------------
InstanceAccel::InstanceAccel()
: InstanceAccel(Type::Instances)
{

}

InstanceAccel::InstanceAccel(Type type)
: m_type(type)
{

}

// ---------------------------------------------------------------------------
void InstanceAccel::build(const Context& ctx, const Instance& instance)
{
    std::vector<OptixInstance> optix_instance(1, static_cast<OptixInstance>(instance));
    build(ctx, optix_instance);
}

void InstanceAccel::build(const Context& ctx, const OptixInstance& instance)
{   
    std::vector<OptixInstance> optix_instance(1, instance);
    build(ctx, optix_instance);
}

void InstanceAccel::build(const Context& ctx, const std::vector<Instance>& instances)
{
    std::vector<OptixInstance> optix_instances(instances.size());
    std::transform(instances.begin(), instances.end(), optix_instances.begin(), 
        [](auto instance){ return static_cast<OptixInstance>(instance); });
    build(ctx, optix_instances);
}

void InstanceAccel::build(const Context& ctx, const std::vector<OptixInstance>& optix_instances)
{
    CUDABuffer<OptixInstance> d_instances; 
    d_instances.copyToDevice(optix_instances);

    OptixBuildInput instance_input = {};
    instance_input.type = static_cast<OptixBuildInputType>(m_type);
    instance_input.instanceArray.instances = d_instances.devicePtr();
    instance_input.instanceArray.numInstances = static_cast<uint32_t>(optix_instances.size());

    m_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        static_cast<OptixDeviceContext>(ctx), 
        &m_options, 
        &instance_input, 
        1,  // num build inputs
        &ias_buffer_sizes ));

    // Allocate buffer to build acceleration structure
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer), 
        ias_buffer_sizes.tempSizeInBytes ));
    CUdeviceptr d_ias_output_buffer; 
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_ias_output_buffer), 
        ias_buffer_sizes.outputSizeInBytes ));
    
    OptixTraversableHandle ias_handle = 0;
    // Build instance AS contains all GASs to describe the scene.
    OPTIX_CHECK(optixAccelBuild(
        static_cast<OptixDeviceContext>(ctx), 
        0,                  // CUDA stream
        &m_options, 
        &instance_input, 
        1,                  // num build inputs
        d_temp_buffer, 
        ias_buffer_sizes.tempSizeInBytes, 
        d_ias_output_buffer, 
        ias_buffer_sizes.outputSizeInBytes, 
        &ias_handle, 
        nullptr,            // emitted property list
        0                   // num emitted properties
    ));

    cuda_free(d_temp_buffer);
    d_instances.free();
}

void InstanceAccel::update(const Context& ctx)
{
    /** Future work */
}

void InstanceAccel::free()
{
    if (d_buffer) cudaFree(reinterpret_cast<void*>(d_buffer));
}

// ---------------------------------------------------------------------------
void InstanceAccel::setFlags(const uint32_t build_flags)
{
    m_options.buildFlags = build_flags;
}

void InstanceAccel::setMotionOptions(const OptixMotionOptions& motion_options)
{
    m_options.motionOptions = motion_options;
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

} // ::oprt