#pragma once

#include <core/util.h>
#include <core/shape.h>
#include <core/material.h>
#include <core/transform.h>
#include <optix/program.h>

namespace pt {

class Primitive {
public:
    Primitive(ShapePtr shape_ptr, MaterialPtr material_ptr, const Transform& transform, uint32_t sbt_index)
    : m_shape_ptr(m_shape_ptr), m_material_ptr(m_material_ptr), m_transform(transform), m_sbt_index(sbt_index) {
        m_program_groups.resize(RAY_TYPE_COUNT);
        for (auto &pg : m_program_groups) {
            pg = ProgramGroup(OPTIX_PROGRAM_GROUP_KIND_HITGROUP);
        }
    }

    // Create programs based on shape type. 
    void create_programs(const OptixDeviceContext& ctx, const OptixModule& module) {
        Assert(!m_program_groups.empty(), "ProgramGroup is not allocated.");
        if (shapetype() == ShapeType::Mesh) {
            m_program_groups[0].create(ctx, ProgramEntry(module, ch_func_str(shape_map[shapetype()]) ) );
            if (m_program_groups.size() > 1)
                m_program_groups[1].create(ctx, ProgramEntry(module, CH_FUNC_STR("occlusion") ) );
        } else {
            m_program_groups[0].create(ctx, ProgramEntry(module, ch_func_str(shape_map[shapetype()]) ), 
                                            ProgramEntry(module, is_func_str(shape_map[shapetype()]) ) );
            if (m_program_groups.size() > 1)
                m_program_groups[1].create(ctx, ProgramEntry(module, CH_FUNC_STR("occlusion") ) );
        }
    }

    // Preparing (alloc and copy) shape data to the device. 
    void prepare_shapedata() { m_shape_ptr->prepare_data(); }

    // Configure the OptixBuildInput from shape data.
    void build_input(OptixBuildInput& bi) { m_shape_ptr->build_input( bi, m_sbt_index ); }

    /** 
     * \brief 
     * Free temporal device side pointers. 
     * \note  
     * Currently, only aabb_buffer is freed
     */
    void free_temp_buffer() { if (m_shape_ptr->type() != ShapeType::Mesh) m_shape_ptr->free_aabb_buffer(); }

    // Bind programs and HitGroupRecord
    void bind_radiance_record(HitGroupRecord record) {
        Assert(!m_program_groups.empty(), "ProgramGroups is not allocated.");
        m_program_groups[0].bind_sbt_and_program(record);
    }
    void bind_occlusion_record(HitGroupRecord record) {
        Assert(m_program_groups.size() > 1, "Occlusion program is not contained in rendering.");
        m_program_groups[1].bind_sbt_and_program(record);
    }

    // Getter 
    uint32_t sbt_index() const { return m_sbt_index; }
    MaterialPtr material() const { return m_material_ptr; }
    ShapePtr shape() const  { return m_shape_ptr; }
    Transform transform() const { return m_transform; }
    ShapeType shapetype() const { return m_shape_ptr->type(); }
    MaterialType materialtype() const { return m_material_ptr->type(); }

    std::vector<ProgramGroup> program_groups() { return m_program_groups; }

private:
    // Member variables.
    ShapePtr m_shape_ptr;
    MaterialPtr m_material_ptr;
    Transform m_transform;

    /** 
     * @param
     * - 0 : for radiance program
     * - 1 : for occlusion program (optional)
     **/
    std::vector<ProgramGroup> m_program_groups;
    // For managing sbt index which associated with a shader.
    uint32_t m_sbt_index { 0 };
};

void build_gas(const OptixDeviceContext& ctx, AccelData& accel_data, const std::vector<Primitive> primitives) {
    std::vector<Primitive> meshes;
    std::vector<Primitive> customs;

    for (auto &p : primitives) {
        if (p.shapetype() == ShapeType::Mesh) meshes.push_back(p);
        else                                  customs.push_back(p);
    }

    auto build_single_gas = [&ctx](std::vector<Primitive> primitives_subset, AccelData::HandleData& handle) {
        if (handle.d_buffer)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(handle.d_buffer)));
            handle.handle = 0;
            handle.d_buffer = 0;
            handle.count = 0;
        }

        handle.count = primitives_subset.size();

        std::vector<OptixBuildInput> build_inputs;
        for (auto &p : primitives_subset) {
            OptixBuildInput build_input;
            p.prepare_shapedata();
            p.build_input(build_input);
            build_inputs.push_back(build_input);
        }

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            ctx,
            &accel_options,
            build_inputs.data(),
            static_cast<unsigned int>(build_inputs.size()),
            &gas_buffer_sizes
        ));

        // temporarily buffer to build AS
        CUdeviceptr d_temp_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

        // Non-compacted output
        CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
            compactedSizeOffset + 8
        ));

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

        OPTIX_CHECK(optixAccelBuild(
            ctx, 
            0, 
            &accel_options,
            build_inputs.data(),
            handle.count,
            d_temp_buffer,
            gas_buffer_sizes.tempSizeInBytes,
            d_buffer_temp_output_gas_and_compacted_size,
            gas_buffer_sizes.outputSizeInBytes,
            &handle.handle,
            &emitProperty,
            1
        ));
        
        // Free temporarily buffers 
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
        /// \note Currently, only aabb buffer is freed.
        for (auto& p : primitives_subset) p.free_temp_buffer(); 

        size_t compacted_gas_size;
        CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

        if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&handle.d_buffer), compacted_gas_size));
            OPTIX_CHECK(optixAccelCompact(ctx, 0, handle.handle, handle.d_buffer, compacted_gas_size, &handle.handle));
            CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
        }
        else {
            handle.d_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }
    };
    
    build_single_gas(meshes, accel_data.meshes);
    build_single_gas(customs, accel_data.customs);
}

void build_ias(const OptixDeviceContext& ctx, 
               const AccelData& accel_data,
               const std::vector<Primitive>& primitives)
{
    
}

}