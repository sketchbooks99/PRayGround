#pragma once

#include <optix.h>
#include <prayground/core/util.h>
#include <prayground/optix/program.h>
#include <prayground/optix/context.h>
#include <prayground/optix/sbt.h>
#include <filesystem>
#include <tuple>

namespace prayground {

class Pipeline {
public:
    Pipeline();
    explicit Pipeline(const std::string& params_name);
    explicit Pipeline(const OptixPipelineCompileOptions& op);
    explicit Pipeline(const OptixPipelineCompileOptions& c_op, const OptixPipelineLinkOptions& l_op);

    // Disallow copy constructor
    Pipeline(const Pipeline& pipeline) = delete;

    explicit operator OptixPipeline() const { return m_pipeline; }
    explicit operator OptixPipeline&() { return m_pipeline; }

    [[nodiscard]] Module createBuiltinIntersectionModule(const Context& ctx, OptixPrimitiveType primitive_type);
        
    [[nodiscard]] Module createModuleFromCudaFile(const Context& ctx, const std::filesystem::path& filename);
    [[nodiscard]] Module createModuleFromCudaSource(const Context& ctx, const std::string& source);

    [[nodiscard]] Module createModuleFromPtxFile(const Context& ctx, const std::filesystem::path& filename);
    [[nodiscard]] Module createModuleFromPtxSource(const Context& ctx, const std::string& source);

    [[nodiscard]] ProgramGroup createRaygenProgram(const Context& ctx, const Module& module, const std::string& func_name);
    [[nodiscard]] ProgramGroup createRaygenProgram(const Context& ctx, const ProgramEntry& entry);
    [[nodiscard]] ProgramGroup createMissProgram(const Context& ctx, const Module& module, const std::string& func_name);
    [[nodiscard]] ProgramGroup createMissProgram(const Context& ctx, const ProgramEntry& entry);
    [[nodiscard]] ProgramGroup createHitgroupProgram(const Context& ctx, const Module& module, const std::string& ch_name);
    [[nodiscard]] ProgramGroup createHitgroupProgram(const Context& ctx, const ProgramEntry& ch_entry);
    [[nodiscard]] ProgramGroup createHitgroupProgram(const Context& ctx, const Module& module, const std::string& ch_name, const std::string& is_name);
    [[nodiscard]] ProgramGroup createHitgroupProgram(const Context& ctx, const ProgramEntry& ch_entry, const ProgramEntry& is_entry);
    [[nodiscard]] ProgramGroup createHitgroupProgram(const Context& ctx, const Module& module, const std::string& ch_name, const std::string& is_name, const std::string& ah_name);
    [[nodiscard]] ProgramGroup createHitgroupProgram(const Context& ctx, const ProgramEntry& ch_entry, const ProgramEntry& is_entry, const ProgramEntry& ah_entry);
    [[nodiscard]] std::pair<ProgramGroup, uint32_t> createCallablesProgram(const Context& ctx, const Module& module, const std::string& dc_name, const std::string& cc_name);
    [[nodiscard]] std::pair<ProgramGroup, uint32_t> createCallablesProgram(const Context& ctx, const ProgramEntry& dc_entry, const ProgramEntry& cc_entry);
    [[nodiscard]] ProgramGroup createExceptionProgram(const Context& ctx, const Module& module, const std::string& func_name);
    [[nodiscard]] ProgramGroup createExceptionProgram(const Context& ctx, const ProgramEntry& entry);

    /** Create pipeline object and calculate the stack sizes of pipeline. */
    void create(const Context& ctx);
    void createFromPrograms(const Context& ctx, const std::vector<ProgramGroup>& prgs);
    void destroy();

    /** Get program groups */
    std::vector<ProgramGroup> programs() const;

    /** Compile options. */
    void setCompileOptions( const OptixPipelineCompileOptions& op );
    void enableMotionBlur();
    void disableMotionBlur();
    void setTraversableGraphFlags(const uint32_t flags);
    void setNumPayloads(const int num_payloads);
    void setNumAttributes(const int num_attributes);
    void setLaunchVariableName(const char* params_name);
    void setExceptionFlags(OptixExceptionFlags flags);
    OptixPipelineCompileOptions compileOptions() const;

    /** Link options */
    void setLinkOptions(const OptixPipelineLinkOptions& op);
#if OPTIX_VERSION < 70700
    // OptixPipelineLinkOptions.debugLevel is deprecated after OptiX 7.7
    void setLinkDebugLevel(const OptixCompileDebugLevel& debug_level);
#endif
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
    
    OptixPipelineCompileOptions m_compile_options {};
    OptixPipelineLinkOptions m_link_options {};
    OptixPipeline m_pipeline { nullptr };
    uint32_t m_trace_depth { 5 }; 
    uint32_t m_cc_depth { 0 }; 
    uint32_t m_dc_depth { 0 };
};

}