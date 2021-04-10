#pragma once 

#include <utility> // for std::pair< , >
#include "../core/util.h"
#include <optix.h>

namespace pt { 

/** OptixModule and the name of entry function */
using ProgramEntry = std::pair<OptixModule, const char*>;

class ProgramGroup {
public: 
    ProgramGroup() {}
    explicit ProgramGroup(OptixProgramGroupKind prg_kind) : m_program_kind(prg_kind), m_program_options({}) {}
    explicit ProgramGroup(OptixProgramGroupKind prg_kind, OptixProgramGroupOptions prg_options)
    : m_program_kind(prg_kind), m_program_options(prg_options) {}

    /** \brief Enable to cast from `ProgramGroup` to `OptixProgramGroup` */
    operator OptixProgramGroup() { return m_program; }

    void destroy() {
        OPTIX_CHECK(optixProgramGroupDestroy(m_program));
    }

    /** \brief create program groups depends on OptixProgramGroupKind */
    template <typename ...Entries>
    void create( const OptixDeviceContext& ctx, Entries... entries)
    {   
        const size_t num_entries = sizeof...(entries); 

        switch(m_program_kind) {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
        case OPTIX_PROGRAM_GROUP_KIND_MISS: 
        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
            if constexpr (num_entries == 1)
                create_single_program(ctx, entries...);
            break;
        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
            if constexpr (num_entries <= 3)
                create_hitgroup_program(ctx, entries...);
            break;
        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
            if constexpr (num_entries == 2)
                create_callable_program(ctx, entries...);
            break;
        }
    }

    /** \brief Creation of a single-call program (Raygen, Miss, Exception) */
    void create_single_program( OptixDeviceContext ctx, 
                                ProgramEntry& entry )
    {
        Assert(m_program_kind == OPTIX_PROGRAM_GROUP_KIND_RAYGEN || 
               m_program_kind == OPTIX_PROGRAM_GROUP_KIND_MISS   || 
               m_program_kind == OPTIX_PROGRAM_GROUP_KIND_EXCEPTION,
               "The OptixProgramGroupKind " + to_str(m_program_kind) + " is not a single-call program." );

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

    /** 
     * \brief Creation of hitgroup programs 
     * \note Only the closest-hit program is used to create hitgroup program. 
     */
    void create_hitgroup_program( OptixDeviceContext ctx, 
                                  ProgramEntry ch_entry) 
    {
        create_hitgroup_program(ctx, ch_entry, ProgramEntry(nullptr, nullptr), ProgramEntry(nullptr, nullptr));
    }
    /** \brief Closest-hit and intersection program are used to create hitgroup program. */
    void create_hitgroup_program( OptixDeviceContext ctx,
                                  ProgramEntry ch_entry,
                                  ProgramEntry is_entry) 
    {
        create_hitgroup_program(ctx, ch_entry, ProgramEntry(nullptr, nullptr), is_entry);
    }
    /** \brief All of programs are used to create hitgroup program. */
    void create_hitgroup_program( OptixDeviceContext ctx,
                                  ProgramEntry ch_entry,
                                  ProgramEntry ah_entry,
                                  ProgramEntry is_entry) 
    {
        Assert(m_program_kind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
               "The OprixProgramGroupKind " + to_str(m_program_kind) + " is not a hitgroup program.");

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

    /** Creation of callable programs */
    void create_callable_program( OptixDeviceContext ctx, 
                                  ProgramEntry dc_entry, 
                                  ProgramEntry cc_entry) 
    {
        Assert(m_program_kind == OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
               "The OptixProgramGroupKind " + to_str(m_program_kind) + " is not a callble program.");
        
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

    template <typename SBTRecord>
    void bind_record(SBTRecord* record) {
        OPTIX_CHECK(optixSbtRecordPackHeader(m_program, record));
    }
private:
    OptixProgramGroup m_program { 0 };
    OptixProgramGroupKind m_program_kind {};
    OptixProgramGroupOptions m_program_options {};
}; 

}