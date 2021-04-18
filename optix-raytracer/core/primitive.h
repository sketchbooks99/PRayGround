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

#include "../core/util.h"
#include "../core/shape.h"
#include "../core/material.h"
#include "../core/texture.h"
#include "../core/transform.h"
#include "../core/cudabuffer.h"
#include "../optix/program.h"
#include "../optix/sbt.h"
#include "../optix/module.h"
#include <algorithm>

namespace oprt {

class Primitive {
public:
    Primitive(Shape* shape_ptr, Material* material_ptr)
    : m_shape_ptr(shape_ptr), m_material_ptr(material_ptr) {
        _init_program_groups();
    }

    Primitive(Shape* shape_ptr, Material* material_ptr, uint32_t sbt_index)
    : m_shape_ptr(shape_ptr), m_material_ptr(material_ptr), m_sbt_index(sbt_index) {
        _init_program_groups();
    }

    // Create programs based on shape type. 
    void create_programs(const OptixDeviceContext& ctx, const OptixModule& module) {
        Assert(!m_program_groups.empty(), "ProgramGroup is not allocated.");
        if (shapetype() == ShapeType::Mesh) {
            // Program for mesh is only a closest-hit program. 
            m_program_groups[0].create( ctx, ProgramEntry( module, ch_func_str( shape_map[shapetype()]).c_str() ) );
            if (m_program_groups.size() > 1) {
                m_program_groups[1].create( ctx, ProgramEntry( module, ch_func_str( shape_occlusion_map[shapetype()]).c_str() ) );
            }
        } else {
            // Programs for custom primitives must include closeset-hit and intersection programs.
            m_program_groups[0].create( ctx, ProgramEntry( module, ch_func_str( shape_map[shapetype()]).c_str() ), 
                                             ProgramEntry( module, is_func_str( shape_map[shapetype()]).c_str() ) );
            if (m_program_groups.size() > 1) {
                m_program_groups[1].create( ctx, ProgramEntry( module, ch_func_str( shape_occlusion_map[shapetype()]).c_str() ),
                                                 ProgramEntry( module, is_func_str( shape_map[shapetype()]).c_str() ) );
                
            }
        
        }
    }

    // Preparing (alloc and copy) shape data to the device. 
    void prepare_shapedata() { m_shape_ptr->prepare_data(); }
    void prepare_matdata() { m_material_ptr->prepare_data(); }

    // Configure the OptixBuildInput from shape data.
    void build_input( OptixBuildInput& bi, unsigned int index_offset ) { m_shape_ptr->build_input( bi, m_sbt_index, index_offset); }

    /** 
     * \brief 
     * Free temporal device side pointers. 
     * \note  
     * Currently, only aabb_buffer is freed
     */
    void free_temp_buffer() { if (m_shape_ptr->type() != ShapeType::Mesh) m_shape_ptr->free_aabb_buffer(); }

    // Bind programs and HitGroupRecord
    template <typename SBTRecord>
    void bind_radiance_record(SBTRecord* record) {
        Assert(!m_program_groups.empty(), "ProgramGroups is not allocated.");
        m_program_groups[0].bind_record(record);
    }
    template <typename SBTRecord>
    void bind_occlusion_record(SBTRecord* record) {
        Assert(m_program_groups.size() > 1, "Occlusion program is not contained in rendering.");
        m_program_groups[1].bind_record(record);
    }

    // Setter
    void set_sbt_index(const uint32_t idx) { m_sbt_index = idx; } 

    // Getter 
    uint32_t sbt_index() const { return m_sbt_index; }
    Material* material() const { return m_material_ptr; }
    Shape* shape() const { return m_shape_ptr; }
    ShapeType shapetype() const { return m_shape_ptr->type(); }
    MaterialType materialtype() const { return m_material_ptr->type(); }

    std::vector<ProgramGroup> program_groups() { return m_program_groups; }

private:
    void _init_program_groups() {
        m_program_groups.resize(RAY_TYPE_COUNT);
        for (auto &pg : m_program_groups) {
            pg = ProgramGroup(OPTIX_PROGRAM_GROUP_KIND_HITGROUP);
        }
    }

    // Member variables.
    Shape* m_shape_ptr;
    Material* m_material_ptr;

    /** 
     * \note
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
 * - Transform stored in this class must not be modified from outside.
 * - I decided to store primitives with the order of meshes -> custom primitves, 
 *   to set the sbt indices in the correct order.
 */
class PrimitiveInstance {
public:
    PrimitiveInstance() : m_transform(Transform()) {}
    explicit PrimitiveInstance(const sutil::Matrix4x4& mat) : m_transform(mat) {}
    explicit PrimitiveInstance(const Transform& transform) : m_transform(transform) {}
    explicit PrimitiveInstance(const Transform& transform, const std::vector<Primitive>& primitives)
    : m_transform(transform), m_primitives(primitives) {}

    void add_primitive(const Primitive& p) { 
        m_primitives.push_back(p); 
        m_primitives.back().set_sbt_index(this->sbt_index_base() + (this->num_primitives() - 1)*RAY_TYPE_COUNT);
    }
    void add_primitive(Shape* shape_ptr, Material* mat_ptr) {
        m_primitives.emplace_back(shape_ptr, mat_ptr);
        m_primitives.back().set_sbt_index(this->sbt_index_base() + (this->num_primitives() - 1)*RAY_TYPE_COUNT);
    }

    /**
     * \brief Sort primitives with the order of meshes -> custom primitives.
     */
    void sort() {
        std::sort(m_primitives.begin(), m_primitives.end(), 
            [](const Primitive& p1, const Primitive& p2){ return (int)p1.shapetype() < (int)p2.shapetype(); });
        uint32_t sbt_index = 0;
        for (auto &p : m_primitives) {
            p.set_sbt_index(this->sbt_index_base() + sbt_index);
            sbt_index += RAY_TYPE_COUNT;
        }
    }

    // Allow to return primitives as lvalue. 
    std::vector<Primitive> primitives() const { return m_primitives; }
    std::vector<Primitive>& primitives() { return m_primitives; }

    size_t num_primitives() const { return m_primitives.size(); }

    void set_sbt_index_base(const unsigned int base) { m_sbt_index_base = base; }
    unsigned int sbt_index_base() const { return m_sbt_index_base; }
    unsigned int sbt_index() const { return m_sbt_index_base + (unsigned int)m_primitives.size() * RAY_TYPE_COUNT; }
    
    void set_transform(const Transform& t) { m_transform = t; } 
    Transform transform() const { return m_transform; }
private:
    Transform m_transform;
    std::vector<Primitive> m_primitives;
    unsigned int m_sbt_index_base { 0 };
};


/**
 * @brief 
 * 
 * @param ctx 
 * @param accel_data 
 * @param ps 
 */
void build_gas(const OptixDeviceContext& ctx, AccelData& accel_data, const PrimitiveInstance& ps) {
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
            cuda_free(handle.d_buffer);
            handle.handle = 0;
            handle.d_buffer = 0;
            handle.count = 0;
        }

        handle.count = static_cast<unsigned int>(primitives_subset.size());

        std::vector<OptixBuildInput> build_inputs(primitives_subset.size());
        unsigned int index_offset = 0;
        for (size_t i=0; i<primitives_subset.size(); i++) {
            primitives_subset[i].prepare_shapedata();
            primitives_subset[i].prepare_matdata();
            primitives_subset[i].build_input(build_inputs[i], index_offset);

            switch ( primitives_subset[i].shapetype() ) {
            case ShapeType::Mesh:
                index_offset += build_inputs[i].triangleArray.numIndexTriplets;
                break;
            case ShapeType::Sphere:
                index_offset += build_inputs[i].customPrimitiveArray.numPrimitives;
                break;
            default:
                break;
            }
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
            0,                      // CUDA stream
            &accel_options,
            build_inputs.data(),
            build_inputs.size(),    
            d_temp_buffer,
            gas_buffer_sizes.tempSizeInBytes,
            d_buffer_temp_output_gas_and_compacted_size,
            gas_buffer_sizes.outputSizeInBytes,
            &handle.handle,
            &emitProperty,
            1                       // num emitted properties
        ));
        
        // Free temporarily buffers 
        cuda_free(d_temp_buffer);
        /// \note Currently, only aabb buffer is freed.
        for (auto& p : primitives_subset) p.free_temp_buffer(); 

        size_t compacted_gas_size;
        CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

        if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&handle.d_buffer), compacted_gas_size));
            OPTIX_CHECK(optixAccelCompact(ctx, 0, handle.handle, handle.d_buffer, compacted_gas_size, &handle.handle));
            cuda_free(d_buffer_temp_output_gas_and_compacted_size);
        }
        else {
            handle.d_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }
    };
    
    if (meshes.size() > 0) build_single_gas(meshes, ps.transform(), accel_data.meshes);
    if (customs.size() > 0) build_single_gas(customs, ps.transform(), accel_data.customs);
}

/**
 * @brief 
 * 
 * @param ctx 
 * @param accel_data 
 * @param primitive_instance 
 * @param sbt_base_offset 
 * @param instance_id 
 * @param instances 
 */
void build_instances(const OptixDeviceContext& ctx, 
               const AccelData& accel_data,
               const PrimitiveInstance& primitive_instance, 
               unsigned int& sbt_base_offset,
               unsigned int& instance_id,
               std::vector<OptixInstance>& instances)
{
    const unsigned int visibility_mask = 255;

    Transform transform = primitive_instance.transform();
    // unsigned int flags = prim_instance.transform().is_identity() 
    //                         ? OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM 
    //                         : OPTIX_INSTANCE_FLAG_NONE;
    unsigned int flags = OPTIX_INSTANCE_FLAG_NONE;

    // Create OptixInstance for the meshes.
    if (accel_data.meshes.handle) {
        OptixInstance instance = { 
            {transform.mat[0], transform.mat[1], transform.mat[2], transform.mat[3],
             transform.mat[4], transform.mat[5], transform.mat[6], transform.mat[7],
             transform.mat[8], transform.mat[9], transform.mat[10], transform.mat[11]},
            instance_id, sbt_base_offset, visibility_mask, flags, 
            accel_data.meshes.handle, /* pad = */ {0, 0}
        };
        sbt_base_offset += (unsigned int)accel_data.meshes.count * RAY_TYPE_COUNT;
        instance_id++;
        instances.push_back(instance);
    }

    // Create OptixInstance for the custom primitives.
    if (accel_data.customs.handle) {
        OptixInstance instance = { 
            {transform.mat[0], transform.mat[1], transform.mat[2], transform.mat[3],
             transform.mat[4], transform.mat[5], transform.mat[6], transform.mat[7],
             transform.mat[8], transform.mat[9], transform.mat[10], transform.mat[11]},
            instance_id, sbt_base_offset, visibility_mask, flags, 
            accel_data.customs.handle, /* pad = */ {0, 0}
        };
        sbt_base_offset += (unsigned int)accel_data.customs.count * RAY_TYPE_COUNT;
        instance_id++;
        instances.push_back(instance);
    }
}

/**
 * @brief Create a material sample programs object
 * 
 * @param ctx 
 * @param module 
 * @param program_groups 
 * @param sbt 
 */
void create_material_sample_programs(
    const OptixDeviceContext& ctx,
    const Module& module, 
    std::vector<ProgramGroup>& program_groups, 
    std::vector<CallableRecord>& callable_records
) {
    // program_groups.clear(); <- Is it needed?

    for (int i = 0; i < (int)MaterialType::Count; i++) 
    {
        // Material type can be queried by iterator.
        program_groups.push_back(ProgramGroup(OPTIX_PROGRAM_GROUP_KIND_CALLABLES));
        callable_records.push_back(CallableRecord());
        program_groups.back().create(
            ctx, 
            ProgramEntry( nullptr, nullptr ), 
            ProgramEntry( (OptixModule)module, cc_func_str( mat_sample_map[ (MaterialType)i ]).c_str() )
        );
        program_groups.back().bind_record(&callable_records.back());
    }
}

void create_texture_eval_programs(
    const OptixDeviceContext& ctx, 
    const Module& module, 
    std::vector<ProgramGroup>& program_groups,
    std::vector<CallableRecord>& callable_records
)
{
    for (int i = 0; i < (int)TextureType::Count; i++)
    {
        program_groups.push_back(ProgramGroup(OPTIX_PROGRAM_GROUP_KIND_CALLABLES));
        callable_records.push_back(CallableRecord());
        program_groups.back().create(
            ctx, 
            ProgramEntry( (OptixModule)module, dc_func_str( tex_eval_map[ (TextureType)i ] ).c_str() ),
            ProgramEntry( nullptr, nullptr )
        );
        program_groups.back().bind_record(&callable_records.back());
    }
}

}