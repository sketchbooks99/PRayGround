#include "program.h"

namespace oprt {

// ---------------------------------------------------------------------------
ProgramGroup::ProgramGroup()
{

}

ProgramGroup::ProgramGroup(const OptixProgramGroupOptions& options) : m_options(options)
{
    
}

// ---------------------------------------------------------------------------
void ProgramGroup::createRaygen(const Context& ctx, const Module& module, const std::string& func_name)
{
    m_kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;

    OptixProgramGroupDesc prog_desc = {};
    char log[2048];
    size_t sizeof_log = sizeof(log);

    prog_desc.kind = m_kind;
    prog_desc.raygen.module = static_cast<OptixModule>(module);
    prog_desc.raygen.entryFunctionName = func_name.c_str();

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        static_cast<OptixDeviceContext>(ctx), 
        &prog_desc, 
        1, 
        &m_options, 
        log, 
        &sizeof_log, 
        &m_program
    ));
}

void ProgramGroup::createRaygen(const Context& ctx, const ProgramEntry& entry)
{
    auto [module, func_name] = entry;
    createRaygen(ctx, module, func_name);
}

// ---------------------------------------------------------------------------
void ProgramGroup::createMiss(const Context& ctx, const Module& module, const std::string& func_name)
{
    m_kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

    OptixProgramGroupDesc prog_desc = {};
    char log[2048];
    size_t sizeof_log = sizeof(log);

    prog_desc.kind = m_kind;
    prog_desc.miss.module = static_cast<OptixModule>(module);
    prog_desc.miss.entryFunctionName = func_name.c_str();

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        static_cast<OptixDeviceContext>(ctx), 
        &prog_desc, 
        1, 
        &m_options, 
        log, 
        &sizeof_log, 
        &m_program
    ));
}

void ProgramGroup::createMiss(const Context& ctx, const ProgramEntry& entry)
{
    auto [module, func_name] = entry;
    createMiss(ctx, module, func_name);
}

// ---------------------------------------------------------------------------
void ProgramGroup::createHitgroup(const Context& ctx, const Module& module, const std::string& ch_name)
{
    createHitgroup(ctx, module, ch_name, "", "");
}

void ProgramGroup::createHitgroup(const Context& ctx, const Module& module, const std::string& ch_name, const std::string& is_name)
{
    createHitgroup(ctx, module, ch_name, is_name, "");
}

void ProgramGroup::createHitgroup(const Context& ctx, const Module& module, const std::string& ch_name, const std::string& is_name, const std::string& ah_name)
{
    ProgramEntry ch_entry{module, ch_name};
    ProgramEntry is_entry{module, is_name};
    ProgramEntry ah_entry{module, ah_name};
    createHitgroup(ctx, ch_entry, is_entry, ah_entry);
}

void ProgramGroup::createHitgroup(const Context& ctx, const ProgramEntry& ch_entry)
{
    ProgramEntry ah_entry{Module(), ""};
    ProgramEntry is_entry{Module(), ""};
    createHitgroup(ctx, ch_entry, is_entry, ah_entry);
}

void ProgramGroup::createHitgroup(const Context& ctx, const ProgramEntry& ch_entry, const ProgramEntry& is_entry)
{
    ProgramEntry ah_entry{Module(), ""};
    createHitgroup(ctx, ch_entry, is_entry, ah_entry);
}

void ProgramGroup::createHitgroup(const Context& ctx, const ProgramEntry& ch_entry, const ProgramEntry& ah_entry, const ProgramEntry& is_entry)
{
    m_kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    char log[2048];
    size_t sizeof_log = sizeof(log);

    auto [ch_module, ch_name] = ch_entry;
    auto [ah_module, ah_name] = ah_entry;
    auto [is_module, is_name] = is_entry;

    OptixProgramGroupDesc prog_desc = {};
    prog_desc.kind = m_kind;
    prog_desc.hitgroup.moduleCH = static_cast<OptixModule>(ch_module);
    prog_desc.hitgroup.entryFunctionNameCH = ch_name == "" ? nullptr : ch_name.c_str();
    prog_desc.hitgroup.moduleAH = static_cast<OptixModule>(ah_module);
    prog_desc.hitgroup.entryFunctionNameAH = ah_name == "" ? nullptr : ah_name.c_str();
    prog_desc.hitgroup.moduleIS = static_cast<OptixModule>(is_module);
    prog_desc.hitgroup.entryFunctionNameIS = is_name == "" ? nullptr : is_name.c_str();
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        static_cast<OptixDeviceContext>(ctx),
        &prog_desc,
        1,
        &m_options,
        log,
        &sizeof_log, 
        &m_program
    ));
}

// ---------------------------------------------------------------------------
void ProgramGroup::createCallables(const Context& ctx, const Module& module, const std::string& dc_name, const std::string& cc_name)
{
    ProgramEntry dc_entry{module, dc_name};
    ProgramEntry cc_entry{module, cc_name};
    createCallables(ctx, dc_entry, cc_entry);
}

void ProgramGroup::createCallables(const Context& ctx, const ProgramEntry& dc_entry, const ProgramEntry& cc_entry) 
{
    m_kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    
    char log[2048];
    size_t sizeof_log = sizeof( log );

    auto [dc_module, dc_name] = dc_entry;
    auto [cc_module, cc_name] = cc_entry;

    OptixProgramGroupDesc prog_desc = {};
    prog_desc.kind = m_kind;
    prog_desc.callables.moduleDC = static_cast<OptixModule>(dc_module);
    prog_desc.callables.entryFunctionNameDC = dc_name == "" ? nullptr : dc_name.c_str();
    prog_desc.callables.moduleCC = static_cast<OptixModule>(cc_module);
    prog_desc.callables.entryFunctionNameCC = cc_name == "" ? nullptr : cc_name.c_str(); 

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        static_cast<OptixDeviceContext>(ctx),
        &prog_desc, 
        1,
        &m_options,
        log,
        &sizeof_log,
        &m_program
    ));   
}

// ---------------------------------------------------------------------------
void ProgramGroup::destroy()
{
    OPTIX_CHECK(optixProgramGroupDestroy(m_program));
}

OptixProgramGroupKind ProgramGroup::kind() const
{
    return m_kind;
}

// ---------------------------------------------------------------------------
OptixProgramGroupOptions ProgramGroup::options() const
{
    return m_options;
}

}
