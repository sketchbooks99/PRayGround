#include "program.h"

namespace oprt {

void ProgramGroup::createSingleProgram( const Context& ctx, const ProgramEntry& entry )
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
        prog_desc.raygen.module = static_cast<OptixModule>(entry.module);
        prog_desc.raygen.entryFunctionName = entry.func_name;
        break;
    case OPTIX_PROGRAM_GROUP_KIND_MISS:
        prog_desc.miss.module = static_cast<OptixModule>(entry.module);
        prog_desc.miss.entryFunctionName = entry.func_name;
        break;
    case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
        prog_desc.exception.module = static_cast<OptixModule>(entry.module);
        prog_desc.exception.entryFunctionName = entry.func_name;
        break;
    default:
        break;
    }

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        static_cast<OptixDeviceContext>(ctx),
        &prog_desc,
        1,
        &m_program_options,
        log, 
        &sizeof_log,
        &m_program
    ));
}

void ProgramGroup::createHitgroupProgram( 
    const Context& ctx, 
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
    prog_desc.hitgroup.moduleCH = static_cast<OptixModule>(ch_entry.module);
    prog_desc.hitgroup.entryFunctionNameCH = ch_entry.func_name;
    prog_desc.hitgroup.moduleAH = static_cast<OptixModule>(ah_entry.module);
    prog_desc.hitgroup.entryFunctionNameAH = ah_entry.func_name;
    prog_desc.hitgroup.moduleIS = static_cast<OptixModule>(is_entry.module);
    prog_desc.hitgroup.entryFunctionNameIS = is_entry.func_name;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        static_cast<OptixDeviceContext>(ctx),
        &prog_desc,
        1,
        &m_program_options,
        log,
        &sizeof_log, 
        &m_program
    ));
}

void ProgramGroup::createCallableProgram( 
    const Context& ctx, 
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
    prog_desc.callables.moduleDC = static_cast<OptixModule>(dc_entry.module);
    prog_desc.callables.entryFunctionNameDC = dc_entry.func_name;
    prog_desc.callables.moduleCC = static_cast<OptixModule>(cc_entry.module);
    prog_desc.callables.entryFunctionNameCC = cc_entry.func_name; 

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        static_cast<OptixDeviceContext>(ctx),
        &prog_desc, 
        1,
        &m_program_options,
        log,
        &sizeof_log,
        &m_program
    ));   
}

ProgramGroup createRayGenProgram( const Context& ctx, const Module& module, const char* entry_name )
{
    ProgramGroup raygen_program(OPTIX_PROGRAM_GROUP_KIND_RAYGEN);
    raygen_program.createSingleProgram( ctx, ProgramEntry( module, entry_name ) );
    return raygen_program;
}

ProgramGroup createMissProgram( const Context& ctx, const Module& module, const char* entry_name )
{
    ProgramGroup miss_program(OPTIX_PROGRAM_GROUP_KIND_MISS);
    miss_program.createSingleProgram( ctx, ProgramEntry( module, entry_name ) );
    return miss_program;
}

}
