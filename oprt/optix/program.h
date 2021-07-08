#pragma once 

#include "../core/util.h"
#include <optix.h>
#include <optix_stubs.h>
#include "module.h"
#include "context.h"

namespace oprt { 

/** OptixModule and the name of entry function */
// using ProgramEntry = std::pair<OptixModule, const char*>;
struct ProgramEntry 
{
    ProgramEntry(const Module& m, const char* fn) : module(m), func_name(fn) {}
    Module module;
    const char* func_name;
};

class ProgramGroup {
public: 
    ProgramGroup() {}
    explicit ProgramGroup(OptixProgramGroupKind prg_kind) : m_program_kind(prg_kind) {}
    explicit ProgramGroup(OptixProgramGroupKind prg_kind, OptixProgramGroupOptions prg_options)
    : m_program_kind(prg_kind), m_program_options(prg_options) {}

    /** @brief Enable to cast from `ProgramGroup` to `OptixProgramGroup` */
    explicit operator OptixProgramGroup() const { return m_program; }
    explicit operator OptixProgramGroup&()      { return m_program; }

    void destroy() {
        OPTIX_CHECK(optixProgramGroupDestroy(m_program));
    }

    /** @brief create program groups depends on OptixProgramGroupKind */
    template <typename ...Entries>
    void create( const Context& ctx, const Entries&... entries)
    {   
        const size_t num_entries = sizeof...(entries); 

        switch(m_program_kind) {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
        case OPTIX_PROGRAM_GROUP_KIND_MISS: 
        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
            if constexpr (num_entries == 1)
                createSingleProgram(ctx, entries...);
            break;
        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
            if constexpr (num_entries <= 3)
                createHitgroupProgram(ctx, entries...);
            break;
        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
            if constexpr (num_entries == 2)
                createCallableProgram(ctx, entries...);
            break;
        }
    }

    /** @brief Creation of a single-call program (Raygen, Miss, Exception) */
    void createSingleProgram( const Context& ctx, 
                                const ProgramEntry& entry );

    /** 
     * @brief Creation of hitgroup programs 
     * @note Only the closest-hit program is used to create hitgroup program. 
     */
    void createHitgroupProgram( const Context& ctx, 
                                const ProgramEntry& ch_entry ) 
    {
        createHitgroupProgram(ctx, ch_entry, ProgramEntry(Module(), nullptr), ProgramEntry(Module(), nullptr));
    }
    /** @brief Closest-hit and intersection program are used to create hitgroup program. */
    void createHitgroupProgram( const Context& ctx,
                                const ProgramEntry& ch_entry,
                                const ProgramEntry& is_entry)
    {
        createHitgroupProgram(ctx, ch_entry, ProgramEntry(Module(), nullptr), is_entry);
    }
    /** @brief All of programs are used to create hitgroup program. */
    void createHitgroupProgram( const Context& ctx,
                                const ProgramEntry& ch_entry,
                                const ProgramEntry& ah_entry,
                                const ProgramEntry& is_entry);

    /** Creation of callable programs */
    void createCallableProgram( const Context& ctx, 
                                  const ProgramEntry& dc_entry, 
                                  const ProgramEntry& cc_entry);

    template <typename SBTRecord>
    void bindRecord(SBTRecord* record) {
        OPTIX_CHECK(optixSbtRecordPackHeader(m_program, record));
    }
private:
    OptixProgramGroup m_program { 0 };
    OptixProgramGroupKind m_program_kind {};
    OptixProgramGroupOptions m_program_options {};
}; 

ProgramGroup createRayGenProgram(const Context& ctx, const Module& module, const char* entry_name);
ProgramGroup createMissProgram(const Context& ctx, const Module& module, const char* entry_name);

}