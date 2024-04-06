#include "geometry_accel.h"
#include <prayground/core/cudabuffer.h>
#include <prayground/core/util.h>
#include <prayground/math/util.h>
#include <algorithm>

namespace prayground {

    // -------------------------------------------------------------
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

    std::vector<std::shared_ptr<Shape>> GeometryAccel::shapes() const 
    {
        return m_shapes;
    }

    // ---------------------------------------------------------------------------
    void GeometryAccel::build(const Context& ctx, CUstream stream)
    {
        ASSERT(m_shapes.size() > 0, "GeometryAccel must have at least one shape.");

        if (d_buffer)
        {
            cuda_free(d_buffer);
            m_handle = 0;
            d_buffer = 0;
            m_count = 0;
        }

        m_build_inputs.resize(m_shapes.size());
        m_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        bool all_type_equal = true;
        for (size_t i = 0; i < m_shapes.size(); i++)
        {
            all_type_equal &= (m_shape_type == m_shapes[i]->type());
            if (!all_type_equal)
            {
                m_build_inputs.clear();
                THROW("All build input types of shapes must be same as type of GeometryAccel.");
            }
            m_build_inputs[i] = m_shapes[i]->createBuildInput();
        }

        if (m_shapes[0]->type() == ShapeType::Curves)
            m_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;

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
        uint32_t num_emit_properties = 0;
        if ((m_options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0)
        {
            emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emit_property.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compacted_size_offset);
            num_emit_properties = 1;
        }

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
            num_emit_properties > 0 ? &emit_property : nullptr, 
            num_emit_properties
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
        ASSERT((m_options.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0, "allowUpdate() must be called before an update operation.");

        m_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

        if (m_build_inputs.size() != m_shapes.size())
            m_build_inputs.resize(m_shapes.size());
        m_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        bool all_type_equal = true;
        for (size_t i = 0; i < m_shapes.size(); i++) {
            all_type_equal &= (m_shape_type == m_shapes[i]->type());
            if (!all_type_equal) {
                m_build_inputs.clear();
                THROW("All build input types of shapes must be same as type of GeometryAccel.");
            }
            m_build_inputs[i] = m_shapes[i]->createBuildInput();
        }

        if (m_shapes[0]->type() == ShapeType::Curves)
            m_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;

        /**
        * @note
        * GASの更新に必要なだけの一時バッファを保存していないので、GASの更新のたびに
        * optixAccelComputeMemoryUsage()を呼ぶ
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
        if (d_buffer) {
            cuda_free(d_buffer);
        }
        d_buffer = 0;
        d_buffer_size = 0;

        m_shapes.clear();
        m_build_inputs.clear();

        // Initialize optix state
        m_handle = 0;
        m_options = {};
        m_count = 0;
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

} // namespace prayground