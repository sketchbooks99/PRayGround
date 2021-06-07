#include "program.h"

namespace oprt {

void ProgramGroup::createSingleProgram( const OptixDeviceContext& ctx, const ProgramEntry& entry )
{
    Assert( m_program_kind == OPTIX_PROGRAM_GROUP_KIND_RAYGEN || 
            m_program_kind == OPTIX_PROGRAM_GROUP_KIND_MISS   || 
            m_program_kind == OPTIX_PROGRAM_GROUP_KIND_EXCEPTION,
            "The OptixProgramGroupKind " + toString(m_program_kind) + " is not a single-call program." );

    OptixProgramGroupDesc prog_desc = {};
    char log[2048];
    size_t sizeof_log = sizeof( log );

    prog_desc.kind = m_program_kind;
    switch(m_program_kind) {
    case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
        prog_desc.raygen.module = entry.first;
        prog_desc.raygen.entryFunctionName = entry.second;
        break;
    case OPTIX_PROGRAM_GROUP_KIND_MISS:
        prog_desc.miss.module = entry.first;
        prog_desc.miss.entryFunctionName = entry.second;
        break;
    case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
        prog_desc.exception.module = entry.first;
        prog_desc.exception.entryFunctionName = entry.second;
        break;
    default:
        break;
    }

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        ctx,
        &prog_desc,
        1,
        &m_program_options,
        log, 
        &sizeof_log,
        &m_program
    ));
}

void ProgramGroup::createHitgroupProgram( 
    const OptixDeviceContext& ctx, 
    const ProgramEntry& ch_entry, 
    const ProgramEntry& ah_entry, 
    const ProgramEntry& is_entry
)
{
    Assert(m_program_kind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
           "The OprixProgramGroupKind " + toString(m_program_kind) + " is not a hitgroup program.");

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupDesc prog_desc = {};
    prog_desc.kind = m_program_kind;
    prog_desc.hitgroup.moduleCH = ch_entry.first;
    prog_desc.hitgroup.entryFunctionNameCH = ch_entry.second;
    prog_desc.hitgroup.moduleAH = ah_entry.first;
    prog_desc.hitgroup.entryFunctionNameAH = ah_entry.second;
    prog_desc.hitgroup.moduleIS = is_entry.first;
    prog_desc.hitgroup.entryFunctionNameIS = is_entry.second;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        ctx,
        &prog_desc,
        1,
        &m_program_options,
        log,
        &sizeof_log, 
        &m_program
    ));
}

void ProgramGroup::createCallableProgram( 
    const OptixDeviceContext& ctx, 
    const ProgramEntry& dc_entry, 
    const ProgramEntry& cc_entry
) 
{
    Assert(m_program_kind == OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
            "The OptixProgramGroupKind " + toString(m_program_kind) + " is not a callble program.");
    
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OptixProgramGroupDesc prog_desc = {};
    prog_desc.kind = m_program_kind;
    prog_desc.callables.moduleDC = dc_entry.first;
    prog_desc.callables.entryFunctionNameDC = dc_entry.second;
    prog_desc.callables.moduleCC = cc_entry.first;
    prog_desc.callables.entryFunctionNameCC = cc_entry.second; 

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        ctx,
        &prog_desc, 
        1,
        &m_program_options,
        log,
        &sizeof_log,
        &m_program
    ));   
}

ProgramGroup createRayGenProgram( const OptixDeviceContext& ctx, const OptixModule& module, const char* entry_name )
{
    ProgramGroup raygen_program(OPTIX_PROGRAM_GROUP_KIND_RAYGEN);
    raygen_program.createSingleProgram( ctx, ProgramEntry( module, entry_name ) );
    return raygen_program;
}

ProgramGroup createMissProgram( const OptixDeviceContext& ctx, const OptixModule& module, const char* entry_name )
{
    ProgramGroup miss_program(OPTIX_PROGRAM_GROUP_KIND_MISS);
    miss_program.createSingleProgram( ctx, ProgramEntry( module, entry_name ) );
    return miss_program;
}

}
