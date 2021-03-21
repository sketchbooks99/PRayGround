/** 
 * \file primitive.h
 * \brief Management of primitives include shape and material.
 * \author Shunji Kiuchi
 * 
 * \details 
 * Primitive has shape, material and its program on CUDA 
 * to describe primitive behaivior during rendering. 
 * PrimitiveInstance has the array of primitives and transform
 * to easily construct Instance AS. 
 */
#pragma once

#include <include/core/util.h>
#include <include/core/shape.h>
#include <include/core/material.h>
#include <include/core/transform.h>
#include <include/optix/program.h>

namespace pt {

class Primitive {
public:
    Primitive(ShapePtr shape_ptr, MaterialPtr material_ptr, uint32_t sbt_index)
    : m_shape_ptr(shape_ptr), m_material_ptr(material_ptr), m_sbt_index(sbt_index) {
        m_program_groups.resize(RAY_TYPE_COUNT);
        for (auto &pg : m_program_groups) {
            pg = ProgramGroup(OPTIX_PROGRAM_GROUP_KIND_HITGROUP);
        }
    }

    // Create programs based on shape type. 
    void create_programs(const OptixDeviceContext& ctx, const OptixModule& module) {
        Assert(!m_program_groups.empty(), "ProgramGroup is not allocated.");
        if (shapetype() == ShapeType::Mesh) {
            // Program for mesh is only a closest-hit program. 
            m_program_groups[0].create(ctx, ProgramEntry(module, ch_func_str(shape_map[shapetype()]) ) );
            if (m_program_groups.size() > 1)
                m_program_groups[1].create(ctx, ProgramEntry(module, CH_FUNC_STR("occlusion") ) );
        } else {
            // Programs for custom primitives must include closeset-hit and intersection programs.
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
    void bind_radiance_record(const HitGroupRecord& record) {
        Assert(!m_program_groups.empty(), "ProgramGroups is not allocated.");
        m_program_groups[0].bind_sbt_and_program(record);
    }
    void bind_occlusion_record(const HitGroupRecord& record) {
        Assert(m_program_groups.size() > 1, "Occlusion program is not contained in rendering.");
        m_program_groups[1].bind_sbt_and_program(record);
    }

    // Getter 
    uint32_t sbt_index() const { return m_sbt_index; }
    MaterialPtr material() const { return m_material_ptr; }
    ShapePtr shape() const  { return m_shape_ptr; }
    ShapeType shapetype() const { return m_shape_ptr->type(); }
    MaterialType materialtype() const { return m_material_ptr->type(); }

    std::vector<ProgramGroup> program_groups() { return m_program_groups; }

private:
    // Member variables.
    ShapePtr m_shape_ptr;
    MaterialPtr m_material_ptr;

    /** 
     * @param
     * - 0 : for radiance program
     * - 1 : for occlusion program (optional)
     **/
    std::vector<ProgramGroup> m_program_groups;
    // For managing sbt index which associated with a shader.
    uint32_t m_sbt_index { 0 };
};

// ---------------------------------------------------------------------
/** 
 * \brief
 * This class store the primitives with same transformation.
 * 
 * \note 
 * Transform stored in this class never be modified from outside, 
 * so transform operations must be performed before constructing 
 * this class.
 */
class PrimitiveInstance {
public:
    PrimitiveInstance() : m_transform(Transform()) {}
    explicit PrimitiveInstance(const Transform& transform) : m_transform(transform) {}
    explicit PrimitiveInstance(const Transform& transform, const std::vector<Primitive>& primitives)
    : m_transform(transform), m_primitives(primitives) {}

    void add_primitive(const Primitive& p) { m_primitives.push_back(p); }
    std::vector<Primitive> primitives() const { return m_primitives; }
    
    void set_transform(const Transform& t) { m_transform = t; } 
    Transform transform() const { return m_transform; }
private:
    Transform m_transform;
    std::vector<Primitive> m_primitives;
};


// ---------------------------------------------------------------------
/** 
 * \brief 
 * Building geometry AS from primitives that share the same transformation.
 * 
 * \note 
 * Call of this funcion :  build_gas(ctx, accel_data, primitive_instances.primitives());
 */ 
void build_gas(OptixDeviceContext& ctx, AccelData& accel_data, const PrimitiveInstance& ps) {
    std::vector<Primitive> meshes;
    std::vector<Primitive> customs;

    for (auto &p : ps.primitives()) {
        if (p.shapetype() == ShapeType::Mesh) meshes.push_back(p);
        else                                  customs.push_back(p);
    }

    auto build_single_gas = [&ctx](std::vector<Primitive> primitives_subset, 
                                   const Transform& transform, 
                                   AccelData::HandleData& handle) 
    {
        if (handle.d_buffer)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(handle.d_buffer)));
            handle.handle = 0;
            handle.d_buffer = 0;
            handle.count = 0;
        }

        Message("build_single_gas() : AccelData handle inited");

        handle.count = primitives_subset.size();

        std::vector<OptixBuildInput> build_inputs(primitives_subset.size());
        for (size_t i=0; i<primitives_subset.size(); i++) {
            if (primitives_subset[i].shapetype() == ShapeType::Mesh) {
                CUDABuffer<float> d_pre_transform;
                float T[12] = {transform.mat[0], transform.mat[1], transform.mat[2], transform.mat[3],
                               transform.mat[4], transform.mat[5], transform.mat[6], transform.mat[7],
                               transform.mat[8], transform.mat[9], transform.mat[10], transform.mat[11]};
                d_pre_transform.alloc_copy(T, sizeof(float)*12);
                build_inputs[i].triangleArray.preTransform = d_pre_transform.d_ptr();
                build_inputs[i].triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
            }
            primitives_subset[i].prepare_shapedata();
            primitives_subset[i].build_input(build_inputs[i]);
        }

        Message("build_single_gas() : Finished to prepare build inputs");

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        Message("Build input size: ", build_inputs.size());

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            ctx,
            &accel_options,
            build_inputs.data(),
            static_cast<int>(build_inputs.size()),
            &gas_buffer_sizes
        ));

        Message("build_single_gas() : Computed the amount of memory for building AS.");

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

        Message("build_single_gas() : Prepare the compacted output.");

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

        Message("build_single_gas() : Builded Acceleration Structure");
        
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

        Message("build_single_gas() : Cleanup temporarily buffers.");
    };
    
    if (meshes.size() > 0) build_single_gas(meshes, ps.transform(), accel_data.meshes);
    if (customs.size() > 0) build_single_gas(customs, ps.transform(), accel_data.customs);
}

// ---------------------------------------------------------------------
void build_ias(const OptixDeviceContext& ctx, 
               const AccelData& accel_data,
               const PrimitiveInstance& prim_instance, 
               const unsigned int sbt_base_offset,
               const unsigned int instance_id,
               std::vector<OptixInstance>& instances)
{
    const unsigned int visibility_mask = 255;

    Transform transform = prim_instance.transform();
    unsigned int sbt_base = sbt_base_offset;
    unsigned int flags = prim_instance.transform().is_identity() 
                            ? OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM 
                            : OPTIX_INSTANCE_FLAG_NONE;

    // Create OptixInstance for the meshes.
    if (accel_data.meshes.handle) {
        OptixInstance instance = { 
            {transform.mat[0], transform.mat[1], transform.mat[2], transform.mat[3],
             transform.mat[4], transform.mat[5], transform.mat[6], transform.mat[7],
             transform.mat[8], transform.mat[9], transform.mat[10], transform.mat[11]},
            instance_id, sbt_base, visibility_mask, flags, 
            accel_data.meshes.handle, {0, 0} /* pad */
        };
        sbt_base += (unsigned int)accel_data.meshes.count;
        instances.push_back(instance);
    }

    // Create OptixInstance for the custom geometry.
    if (accel_data.customs.handle) {
        OptixInstance instance = { 
            {transform.mat[0], transform.mat[1], transform.mat[2], transform.mat[3],
             transform.mat[4], transform.mat[5], transform.mat[6], transform.mat[7],
             transform.mat[8], transform.mat[9], transform.mat[10], transform.mat[11]},
            instance_id, sbt_base, visibility_mask, flags, 
            accel_data.customs.handle, {0, 0} /* pad */
        };
        instances.push_back(instance);
    }
}

}