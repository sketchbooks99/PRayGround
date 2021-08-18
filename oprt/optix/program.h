#pragma once 

#include <optix.h>
#include <optix_stubs.h>
#include <oprt/optix/macros.h>
#include <utility>
#include "module.h"
#include "context.h"

namespace oprt { 

/** OptixModule and the name of entry function */
using ProgramEntry = std::pair<Module, std::string>;

class ProgramGroup {
public: 
    ProgramGroup();
    explicit ProgramGroup(const OptixProgramGroupOptions& options);

    /** @brief Enable to cast from `ProgramGroup` to `OptixProgramGroup` */
    explicit operator OptixProgramGroup() const { return m_program; }
    explicit operator OptixProgramGroup&()      { return m_program; }

    void createRaygen(const Context& ctx, const Module& module, const std::string& func_name);
    void createRaygen(const Context& ctx, const ProgramEntry& entry);
    void createMiss(const Context& ctx, const Module& module, const std::string& func_name);
    void createMiss(const Context& ctx, const ProgramEntry& entry);
    void createHitgroup(const Context& ctx, const Module& module, const std::string& ch_name);
    void createHitgroup(const Context& ctx, const ProgramEntry& ch_entry);
    void createHitgroup(const Context& ctx, const Module& module, const std::string& ch_name, const std::string& is_name);
    void createHitgroup(const Context& ctx, const ProgramEntry& ch_entry, const ProgramEntry& is_entry);
    void createHitgroup(const Context& ctx, const Module& module, const std::string& ch_name, const std::string& is_name, const std::string& ah_name);
    void createHitgroup(const Context& ctx, const ProgramEntry& ch_entry, const ProgramEntry& is_entry, const ProgramEntry& ah_entry);
    void createException(const Context& ctx, const Module& module, const std::string& func_name);
    void createException(const Context& ctx, const ProgramEntry& entry);
    [[nodiscard]] int32_t createCallables(const Context& ctx, const Module& module, const std::string& dc_name, const std::string& cc_name);
    [[nodiscard]] int32_t createCallables(const Context& ctx, const ProgramEntry& dc_entry, const ProgramEntry& cc_entry);

    void destroy();

    template <typename SBTRecord>
    void bindRecord(SBTRecord* record) {
        OPTIX_CHECK(optixSbtRecordPackHeader(m_program, record));
    }

    OptixProgramGroupKind kind() const;
    OptixProgramGroupOptions options() const;
private:
    OptixProgramGroup m_program { 0 };
    OptixProgramGroupKind m_kind {};
    OptixProgramGroupOptions m_options {};
}; 

}