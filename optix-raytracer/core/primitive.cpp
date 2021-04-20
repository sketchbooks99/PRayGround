#include "primitive.h"
#include "../core/texture.h"
#include <optix_stubs.h>

namespace oprt {

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
            primitives_subset[i].build_input(build_inputs[i]);

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
 * @param callable_records 
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

/**
 * @brief Create a texture eval programs object
 * 
 * @param ctx 
 * @param module 
 * @param program_groups 
 * @param callable_records 
 */
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