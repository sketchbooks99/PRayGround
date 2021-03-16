#pragma once 

#include <core/scene.h>
#include <optix/sbt.h>

namespace pt {

void Scene::create_hitgroup_programs(const OptixDeviceContext& ctx, const OptixModule& module) {
    for (auto& p : m_primitives)
        p.create_programs(ctx, module);
}

void Scene::build_gas(const OptixDeviceContext& ctx, AccelData& accel_data) {
    std::vector<Primitive> meshes;
    std::vector<Primitive> customs;

    for (auto &p : m_primitives) {
        if (p.shapetype() == ShapeType::Mesh) meshes.push_back(p);
        else                                      customs.push_back(p);
    }

    auto build_single_gas = [&ctx](std::vector<Primitive> primitives, AccelData::HandleData& handle) {
        if (handle.d_buffer)
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(handle.d_buffer)));
            handle.handle = 0;
            handle.d_buffer = 0;
            handle.count = 0;
        }

        handle.count = primitives.size();

        std::vector<OptixBuildInput> build_inputs;
        for (auto &p : primitives) {
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
        for (auto& p : primitives) p.free_temp_buffer(); 

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

void Scene::create_hitgroup_sbt(const OptixModule& module, OptixShaderBindingTable& sbt) {
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    std::vector<HitGroupRecord> hitgroup_records(RAY_TYPE_COUNT * m_primitives.size());
    int sbt_idx = 0;
    for (auto& p : m_primitives) {
        // Bind HitGroupData to radiance program. 
        hitgroup_records[sbt_idx].data.shapedata = p.shape()->get_dptr();
        hitgroup_records[sbt_idx].data.matptr = (MaterialPtr)p.material()->get_dptr();
        p.bind_radiance_record(hitgroup_records[sbt_idx]);

        // Bind HitGroupData to occlusion program. 
        sbt_idx++;
        memset(&hitgroup_records[sbt_idx], 0, hitgroup_record_size);
        p.bind_occlusion_record(hitgroup_records[sbt_idx]);
    }
}

}