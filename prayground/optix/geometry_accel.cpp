#include "geometry_accel.h"
#include <prayground/core/cudabuffer.h>
#include <prayground/core/util.h>
#include <algorithm>

namespace prayground {

// GeometryAccel -------------------------------------------------------------
GeometryAccel::GeometryAccel(ShapeType shape_type)
: m_shape_type(shape_type)
{
    m_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
}

GeometryAccel::~GeometryAccel()
{

}

// ---------------------------------------------------------------------------
void GeometryAccel::addShape(const std::shared_ptr<Shape>& shape)
{
    m_shapes.push_back(shape);
    m_count++;
}

// ---------------------------------------------------------------------------
void GeometryAccel::build(const Context& ctx, CUstream stream)
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

    bool all_type_equal = true;
    for (size_t i = 0; i < m_shapes.size(); i++)
    {
        all_type_equal &= (m_shape_type == m_shapes[i]->type());
        if (!all_type_equal)
        {
            m_build_inputs.clear();
            Throw("All build input types of shapes must be same as type of GeometryAccel.");
        }
        m_build_inputs[i] = m_shapes[i]->createBuildInput();
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
        stream, 
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

    cuda_free(d_temp_buffer);

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

void GeometryAccel::update(const Context& ctx, CUstream stream)
{
    Assert((m_options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0, "prayground::GeometryAccel::update(): allowUpdate() must be called when using update operation.");

    m_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    /**
     * @note
     * - 自分自身がGASをアップデートするための一時的なバッファを保持していなかった場合，
     *   optixAccelBuild() を呼ぶためのメモリ領域を再計算してGASの更新を行うのに
     *   必要なだけのバッファを確保する
     * 
     * - is_hold_temp_buffer のフラグで更新時に必要なメモリ量を計算するか判断すると
     *   d_temp_buffer よりも真に必要なメモリ量が多かった場合にクラッシュする。
     *   そこまでコストが高くないなら、optixAccelComputeMemoryUsage()で
     *   毎度メモリ量を計算するのが安全？
     */
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        static_cast<OptixDeviceContext>(ctx), 
        &m_options, 
        m_build_inputs.data(),
        static_cast<uint32_t>(m_build_inputs.size()), 
        &gas_buffer_sizes
    ));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempUpdateSizeInBytes));
    size_t d_temp_buffer_size = gas_buffer_sizes.tempUpdateSizeInBytes;

    OPTIX_CHECK(optixAccelBuild(
        static_cast<OptixDeviceContext>(ctx),
        stream,
        &m_options,
        m_build_inputs.data(),
        m_build_inputs.size(),
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

void GeometryAccel::free()
{
    if (d_buffer) cuda_free(d_buffer);
    d_buffer_size = 0;
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

// ---------------------------------------------------------------------------
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

bool GeometryAccel::isBuilded() const
{
    return (bool)d_buffer;
}

} // ::prayground