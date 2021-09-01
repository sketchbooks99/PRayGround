#include "accel.h"
#include <prayground/core/cudabuffer.h>
#include <algorithm>

namespace prayground {

// GeometryAccel -------------------------------------------------------------
GeometryAccel::GeometryAccel(Type type)
: m_type(type)
{
    m_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
}

GeometryAccel::~GeometryAccel()
{

}

// ---------------------------------------------------------------------------
void GeometryAccel::addShape(const std::shared_ptr<Shape>& shape)
{
    Assert(shape->buildInputType() == static_cast<OptixBuildInputType>(m_type),
        "prayground::GeometryAccel::addShape(): The type of shape must be same as the type of GeometryAccel.");
    m_shapes.emplace_back(shape);
}

std::shared_ptr<Shape> GeometryAccel::shapeAt(const int idx) const
{
    return m_shapes[idx];
}

// ---------------------------------------------------------------------------
void GeometryAccel::build(const Context& ctx)
{
    if (m_shapes.size() == 0)
    {
        Message(MSG_ERROR, "prayground::GeoetryAccel::build(): The number of shapes is 0.");
        return;
    }

    if (d_buffer)
    {
        cuda_free(d_buffer);
        m_handle = 0;
        d_buffer = 0; 
        m_count = 0;
    }

    std::vector<OptixBuildInput> m_build_inputs(m_shapes.size());
    m_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    for (size_t i = 0; i < m_shapes.size(); i++)
    {
        m_shapes[i]->buildInput(m_build_inputs[i]);
    }

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        static_cast<OptixDeviceContext>(ctx), 
        &m_options, 
        m_build_inputs.data(),
        static_cast<uint32_t>(m_build_inputs.size()), 
        &gas_buffer_sizes
    ));

    // Temporarily buffer to build GAS
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
        m_build_inputs.data(), 
        m_build_inputs.size(), 
        d_temp_buffer, 
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes, 
        &m_handle, 
        &emit_property, 
        1
    ));

    d_buffer_size = gas_buffer_sizes.outputSizeInBytes;

    if (is_hold_temp_buffer)
    {
        d_temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;
    }
    else
    {
        cuda_free(d_temp_buffer);
        d_temp_buffer_size = 0;
    }

    if ((m_options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0)
    {
        size_t compacted_gas_size;
        CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emit_property.result, sizeof(size_t), cudaMemcpyDeviceToHost));

        if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer), compacted_gas_size));
            OPTIX_CHECK(optixAccelCompact(static_cast<OptixDeviceContext>(ctx), 0, m_handle, d_buffer, compacted_gas_size, &m_handle));
            cuda_free(d_buffer_temp_output_gas_and_compacted_size);
            d_buffer_size = compacted_gas_size;
        }
        else {
            d_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }
    }
}

void GeometryAccel::update(const Context& ctx)
{
    if ((m_options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) == 0)
    {
        Message(MSG_ERROR, "prayground::GeometryAccel::update(): Update of this geometry accel is not allowed. Please enable update by gas.allowUpdate()");
        return;
    }

    m_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    m_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    OptixAccelBufferSizes gas_buffer_sizes;

    /**
     * @note
     * 自分自身がGASをアップデートするための一時的なバッファを保持していなかった場合，
     * optixAccelBuild() を呼ぶためのメモリ領域を再計算する
     */
    if (!d_temp_buffer)
    {
        /**
         * @note 
         * GASに含まれるOptixBuildInput を全て作り直すのは手間な気がする．
         * どのbuildInputが更新されたかを判定する必要があって，それはそれで面倒な気がする．
         * 各buildInput をShape単位で管理するようにすれば、GASが管理せずとも
         * 頂点やShader binding table のインデックス周りの変更はShape単位で行える
         * buildInputの性質上、頂点情報やAABB、Shader binding table のインデックス等は
         * デバイス側のポインタに紐づけられているので関節的に変更が可能だと思われる
         */
        m_build_inputs.resize(m_shapes.size());

        for (size_t i = 0; i < m_shapes.size(); i++)
        {
            m_shapes[i]->buildInput(m_build_inputs[i]);
        }

        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            static_cast<OptixDeviceContext>(ctx), 
            &m_options, 
            m_build_inputs.data(),
            static_cast<uint32_t>(m_build_inputs.size()), 
            &gas_buffer_sizes
        ));
    }
}

void GeometryAccel::free()
{
    if (d_buffer) cudaFree(reinterpret_cast<void*>(d_buffer));
    if (d_temp_buffer) cudaFree(reinterpret_cast<void*>(d_temp_buffer));
    d_buffer_size = 0;
    d_temp_buffer_size = 0;
}

// ---------------------------------------------------------------------------
void GeometryAccel::enableHoldTempBuffer()
{
    is_hold_temp_buffer = true;
}

void GeometryAccel::disableHoldTempBuffer()
{
    is_hold_temp_buffer = false;
}

// ---------------------------------------------------------------------------
void GeometryAccel::setFlags(const uint32_t build_flags)
{
    m_options.buildFlags = build_flags;
}

void GeometryAccel::allowUpdate()
{
    m_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
}

void GeometryAccel::allowCompaction()
{
    m_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
}

void GeometryAccel::preferFastTrace()
{
    m_options.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
}

void GeometryAccel::preferFastBuild()
{
    m_options.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
}

void GeometryAccel::allowRandomVertexAccess()
{
    m_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
}

void GeometryAccel::disableUpdate()
{
    m_options.buildFlags &= ~OPTIX_BUILD_FLAG_ALLOW_UPDATE;
}

void GeometryAccel::disableCompaction()
{
    m_options.buildFlags &= ~OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
}

void GeometryAccel::disableFastTrace()
{
    m_options.buildFlags &= ~OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
}

void GeometryAccel::disableFastBuild()
{
    m_options.buildFlags &= ~OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
}

void GeometryAccel::disableRandomVertexAccess()
{
    m_options.buildFlags &= ~OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
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

CUdeviceptr GeometryAccel::deviceBuffer() const
{
    return d_buffer;
}

size_t GeometryAccel::deviceBufferSize() const
{
    return d_buffer_size;
}

CUdeviceptr GeometryAccel::deviceTempBuffer() const
{
    return d_temp_buffer;
}

size_t GeometryAccel::deviceTempBufferSize() const
{
    return d_temp_buffer_size;
}

// InstanceAccel -------------------------------------------------------------
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
    m_instances.emplace_back(instance);
}

Instance& InstanceAccel::instanceAt(const int32_t idx)
{
    return m_instances[idx];
}

// ---------------------------------------------------------------------------
void InstanceAccel::build(const Context& ctx)
{
    CUDABuffer<OptixInstance> d_instances;
    std::vector<OptixInstance> optix_instances;
    std::transform(m_instances.begin(), m_instances.end(), std::back_inserter(optix_instances),
        [](const Instance& instance) { return static_cast<OptixInstance>(instance);  });
    d_instances.copyToDevice(optix_instances);

    OptixBuildInput instance_input = {};
    instance_input.type = static_cast<OptixBuildInputType>(m_type);
    instance_input.instanceArray.instances = d_instances.devicePtr();
    instance_input.instanceArray.numInstances = static_cast<uint32_t>(m_instances.size());

    m_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        static_cast<OptixDeviceContext>(ctx), 
        &m_options, 
        &instance_input, 
        1,  // num build inputs
        &ias_buffer_sizes ));

    // Allocate buffer to build acceleration structure
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer), 
        ias_buffer_sizes.tempSizeInBytes ));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer), 
        ias_buffer_sizes.outputSizeInBytes ));
    
    // Build instance AS contains all GASs to describe the scene.
    OPTIX_CHECK(optixAccelBuild(
        static_cast<OptixDeviceContext>(ctx), 
        0,                  // CUDA stream
        &m_options, 
        &instance_input, 
        1,                  // num build inputs
        d_temp_buffer, 
        ias_buffer_sizes.tempSizeInBytes, 
        d_buffer, 
        ias_buffer_sizes.outputSizeInBytes, 
        &m_handle, 
        nullptr,            // emitted property list
        0                   // num emitted properties
    ));

    if (is_hold_temp_buffer)
        d_temp_buffer_size = ias_buffer_sizes.tempSizeInBytes;
    else
    {
        cuda_free(d_temp_buffer);
        d_temp_buffer_size = 0;
    }
    /// @note Is this release of pointer needed?
    d_instances.free();  
}

void InstanceAccel::update(const Context& ctx)
{
    TODO_MESSAGE();
}

void InstanceAccel::free()
{
    if (d_buffer) cudaFree(reinterpret_cast<void*>(d_buffer));
    if (d_temp_buffer) cudaFree(reinterpret_cast<void*>(d_temp_buffer));
    d_buffer_size = 0;
    d_temp_buffer_size = 0;
}

// ---------------------------------------------------------------------------
void InstanceAccel::enableHoldTempBuffer()
{
    is_hold_temp_buffer = true;
}

void InstanceAccel::disableHoldTempBuffer()
{
    is_hold_temp_buffer = false;
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

CUdeviceptr InstanceAccel::deviceTempBuffer() const
{
    return d_temp_buffer;
}

size_t InstanceAccel::deviceTempBufferSize() const
{
    return d_temp_buffer_size;
}

} // ::prayground