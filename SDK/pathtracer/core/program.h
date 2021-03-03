#pragma once 

#include <optix.h>

#include "core_util.h"

namespace pt { 

class Program {
public: 
    explicit Program(OptixProgramGroupKind prg_kind) : m_program_kind(prg_kind) {}

    void create(const OptixDeviceContext& ctx, 
                const OptixModule& module, 
                const std::string& entry_func_str)
    {
        OptixProgramGroupDesc prog_desc = {};
        prog_desc.kind = m_program_kind;
        char log[2048];
        size_t sizeof_log = sizeof( log );

        switch(m_program_kind) {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
            prog_desc.raygen.module = module;
            prog_desc.raygen.entryFunctionName = entry_func_str;
        case OPTIX_PROGRAM_GROUP_KIND_MISS:
            prog_desc.miss.module = module;
            prog_desc.miss.entryFunctionName = entry_func_str;
        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
            prog_desc.exception.module = module;
            prog_desc.exception.entryFunctionName = entry_func_str;
        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
        }
        prog_desc.
    }

    // allocate and copy data from host to device.
    template <typename SBTRecord>
    void bind_sbtrecord(SBTRecord record) {
        OPTIX_CHECK(optixSbtRecordPackHeader(m_program, &m_record));

        CUdeviceptr d_records = 0;
        const size_t record_size = sizeof(SBTRecord);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&d_records),
            record_size
        ));

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_records),
            &m_record,
            record_size,
            cudaMemcpyHostToDevice
        ));
    }

    // Get program group
    OptixProgramGroup get_program_group() { return m_program_group; }

private:
    OptixProgramGroup m_program_group;
    OptixProgramGroupKind m_program_kind;
}; 

}