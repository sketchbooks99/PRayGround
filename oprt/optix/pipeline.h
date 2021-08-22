#pragma once

#include <optix.h>
#include <oprt/core/util.h>
#include <oprt/optix/program.h>
#include <oprt/optix/context.h>
#include <oprt/optix/sbt.h>
#include <filesystem>
#include <tuple>

namespace oprt {

/**
 * @note
 * template <class RaygenRecord, class MissRecord, class HitgroupRecord, 
 *         class CallablesRecord, class ExceptionRecord, unsigned int NRay>
 * class Pipeline;
 * こんな感じにして内部にSBTを持つと楽？ただ、こうすると実装をヘッダファイルに
 * 全て書く必要があってすごくやりたくない ...
 * 
 * とりあえずはbindRaygenRecord的な関数だけtemplateで実装して、コーディング側で注意するようにする
 */

class Pipeline {
public:
    Pipeline();
    explicit Pipeline(const std::string& params_name);
    explicit Pipeline(const OptixPipelineCompileOptions& op);
    explicit Pipeline(const OptixPipelineCompileOptions& c_op, const OptixPipelineLinkOptions& l_op);

    explicit operator OptixPipeline() const { return m_pipeline; }
    explicit operator OptixPipeline&() { return m_pipeline; }

    [[nodiscard]] Module createModuleFromCudaFile(const Context& ctx, const std::filesystem::path& filename);
    [[nodiscard]] Module createModuleFromCudaSource(const Context& ctx, const std::string& source);

    [[nodiscard]] Module createModuleFromPtxFile(const Context& ctx, const std::filesystem::path& filename);
    [[nodiscard]] Module createModuleFromPtxSource(const Context& ctx, const std::string& source);

    void createRaygenProgram(const Context& ctx, const Module& module, const std::string& func_name);
    void createRaygenProgram(const Context& ctx, const ProgramEntry& entry);
    void createMissProgram(const Context& ctx, const Module& module, const std::string& func_name);
    void createMissProgram(const Context& ctx, const ProgramEntry& entry);
    void createHitgroupProgram(const Context& ctx, const Module& module, const std::string& ch_name);
    void createHitgroupProgram(const Context& ctx, const ProgramEntry& ch_entry);
    void createHitgroupProgram(const Context& ctx, const Module& module, const std::string& ch_name, const std::string& is_name);
    void createHitgroupProgram(const Context& ctx, const ProgramEntry& ch_entry, const ProgramEntry& is_entry);
    void createHitgroupProgram(const Context& ctx, const Module& module, const std::string& ch_name, const std::string& is_name, const std::string& ah_name);
    void createHitgroupProgram(const Context& ctx, const ProgramEntry& ch_entry, const ProgramEntry& is_entry, const ProgramEntry& ah_entry);
    uint32_t createCallablesProgram(const Context& ctx, const Module& module, const std::string& dc_name, const std::string& cc_name);
    uint32_t createCallablesProgram(const Context& ctx, const ProgramEntry& dc_entry, const ProgramEntry& cc_entry);
    void createExceptionProgram(const Context& ctx, const Module& module, const std::string& func_name);
    void createExceptionProgram(const Context& ctx, const ProgramEntry& entry);

    template <class SBTRecord>
    void bindRaygenRecord(SBTRecord* record)
    {
        Assert((OptixProgramGroup)m_raygen_program, "Raygen program has not been create yet.");
        m_raygen_program.bindRecord(record);
    }

    template <class SBTRecord>
    void bindMissRecord(SBTRecord* record, const int idx)
    {
        Assert(idx < m_miss_programs.size(), "Out of ranges of miss programs");
        m_miss_programs[idx].bindRecord(record);
    }

    template <class SBTRecord>
    void bindHitgroupRecord(SBTRecord* record, const int idx)
    {
        Assert(idx < m_hitgroup_programs.size(), "Out of ranges of hitgroup programs");
        m_hitgroup_programs[idx].bindRecord(record);
    }

    template <class SBTRecord>
    void bindCallablesRecord(SBTRecord* record, const int idx)
    {
        Assert(idx < m_callables_programs.size(), "Out of ranges of miss programs");
        m_callables_programs[idx].bindRecord(record);
    }

    template <class SBTRecord>
    void bindExceptionRecord(SBTRecord* record)
    {
        Assert((OptixProgramGroup)m_exception_program, "Exceptino program has not been create yet.");
        m_exception_program.bindRecord(record);
    }

    /** Create pipeline object and calculate the stack sizes of pipeline. */
    void create(const Context& ctx);
    void destroy();

    /** Compile options. */
    void setCompileOptions( const OptixPipelineCompileOptions& op );
    void usesMotionBlur(const bool is_use);
    void setTraversableGraphFlags(const uint32_t flags);
    void setNumPayloads(const int num_payloads);
    void setNumAttributes(const int num_attributes);
    void setLaunchVariableName(const std::string& params_name);
    void setExceptionFlags(const OptixExceptionFlags& flags);
    OptixPipelineCompileOptions compileOptions() const;

    /** Link options */
    void setLinkOptions(const OptixPipelineLinkOptions& op);
    void setLinkDebugLevel( const OptixCompileDebugLevel& debug_level );
    OptixPipelineLinkOptions linkOptions() const;

    /** Depth of traversal */
    void setTraceDepth(const uint32_t depth);
    uint32_t traceDepth() const;

    /** Depth of continuation-callable */
    void setContinuationCallableDepth(const uint32_t depth);
    uint32_t continuationCallableDepth() const;

    /** Depth of direct-callable */
    void setDirectCallableDepth(const uint32_t depth);
    uint32_t directCallableDepth() const;

private:
    void _initCompileOptions();
    void _initLinkOptions();

    std::vector<Module> m_modules;

    ProgramGroup m_raygen_program;
    std::vector<ProgramGroup> m_miss_programs;
    std::vector<ProgramGroup> m_hitgroup_programs;
    std::vector<ProgramGroup> m_callables_programs;
    ProgramGroup m_exception_program;
    
    OptixPipelineCompileOptions m_compile_options = {};
    OptixPipelineLinkOptions m_link_options = {};
    OptixPipeline m_pipeline { nullptr };
    uint32_t m_trace_depth { 5 }; 
    uint32_t m_cc_depth { 0 }; 
    uint32_t m_dc_depth { 0 };

};

}