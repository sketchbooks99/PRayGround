#include "pipeline.h"
#include <prayground_config.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

namespace prayground {

// --------------------------------------------------------------------
Pipeline::Pipeline()
{
    _initCompileOptions();
    _initLinkOptions();
}

Pipeline::Pipeline(const std::string& params_name)
{
    _initCompileOptions();
    m_compile_options.pipelineLaunchParamsVariableName = params_name.c_str();
    _initLinkOptions();
}

Pipeline::Pipeline(const OptixPipelineCompileOptions& c_op)
    : m_compile_options(c_op)
{
    _initLinkOptions();
}

Pipeline::Pipeline(const OptixPipelineCompileOptions& c_op, const OptixPipelineLinkOptions& l_op)
    : m_compile_options(c_op), m_link_options(l_op)
{

}

// --------------------------------------------------------------------
[[nodiscard]] Module Pipeline::createModuleFromCudaFile(const Context& ctx, const std::filesystem::path& filename)
{
#if !(CUDA_NVRTC_ENABLED)
    static_assert(false);
#endif
    m_modules.emplace_back(Module{});
    m_modules.back().createFromCudaFile(ctx, filename, m_compile_options);
    return m_modules.back();
}

[[nodiscard]] Module Pipeline::createModuleFromCudaSource(const Context& ctx, const std::string& source)
{
#if !(CUDA_NVRTC_ENABLED)
    static_assert(false);
#endif
    m_modules.emplace_back(Module{});
    m_modules.back().createFromCudaSource(ctx, source, m_compile_options);
    return m_modules.back();
}

[[nodiscard]] Module Pipeline::createModuleFromPtxFile(const Context& ctx, const std::filesystem::path& filename)
{
    TODO_MESSAGE();
    return Module{};
}

[[nodiscard]] Module Pipeline::createModuleFromPtxSource(const Context& ctx, const std::string& source)
{
    TODO_MESSAGE();
    return Module{};
}

// --------------------------------------------------------------------
[[nodiscard]]
ProgramGroup Pipeline::createRaygenProgram(const Context& ctx, const Module& module, const std::string& func_name)
{
    m_raygen_program.createRaygen(ctx, module, func_name);
    return m_raygen_program;
}

[[nodiscard]]
ProgramGroup Pipeline::createRaygenProgram(const Context& ctx, const ProgramEntry& entry)
{
    m_raygen_program.createRaygen(ctx, entry);
    return m_raygen_program;
}

[[nodiscard]]
ProgramGroup Pipeline::createMissProgram(const Context& ctx, const Module& module, const std::string& func_name)
{
    m_miss_programs.emplace_back(ProgramGroup{});
    m_miss_programs.back().createMiss(ctx, module, func_name);
    return m_miss_programs.back();
}

[[nodiscard]]
ProgramGroup Pipeline::createMissProgram(const Context& ctx, const ProgramEntry& entry)
{
    m_miss_programs.emplace_back(ProgramGroup{});
    m_miss_programs.back().createMiss(ctx, entry);
    return m_miss_programs.back();
}

[[nodiscard]]
ProgramGroup Pipeline::createHitgroupProgram(const Context& ctx, const Module& module, const std::string& ch_name)
{
    m_hitgroup_programs.emplace_back(ProgramGroup{});
    m_hitgroup_programs.back().createHitgroup(ctx, module, ch_name);
    return m_hitgroup_programs.back();
}

[[nodiscard]]
ProgramGroup Pipeline::createHitgroupProgram(const Context& ctx, const ProgramEntry& ch_entry)
{
    m_hitgroup_programs.emplace_back(ProgramGroup{});
    m_hitgroup_programs.back().createHitgroup(ctx, ch_entry);
    return m_hitgroup_programs.back();
}
[[nodiscard]]
ProgramGroup Pipeline::createHitgroupProgram(const Context& ctx, const Module& module, const std::string& ch_name, const std::string& is_name)
{
    m_hitgroup_programs.emplace_back(ProgramGroup{});
    m_hitgroup_programs.back().createHitgroup(ctx, module, ch_name, is_name);
    return m_hitgroup_programs.back();
}

[[nodiscard]]
ProgramGroup Pipeline::createHitgroupProgram(const Context& ctx, const ProgramEntry& ch_entry, const ProgramEntry& is_entry)
{
    m_hitgroup_programs.emplace_back(ProgramGroup{});
    m_hitgroup_programs.back().createHitgroup(ctx, ch_entry, is_entry);
    return m_hitgroup_programs.back();
}
[[nodiscard]]
ProgramGroup Pipeline::createHitgroupProgram(const Context& ctx, const Module& module, const std::string& ch_name, const std::string& is_name, const std::string& ah_name)
{
    m_hitgroup_programs.emplace_back(ProgramGroup{});
    m_hitgroup_programs.back().createHitgroup(ctx, module, ch_name, is_name, ah_name);
    return m_hitgroup_programs.back();
}
[[nodiscard]]
ProgramGroup Pipeline::createHitgroupProgram(const Context& ctx, const ProgramEntry& ch_entry, const ProgramEntry& is_entry, const ProgramEntry& ah_entry)
{
    m_hitgroup_programs.emplace_back(ProgramGroup{});
    m_hitgroup_programs.back().createHitgroup(ctx, ch_entry, is_entry, ah_entry);
    return m_hitgroup_programs.back();
}

[[nodiscard]]
std::pair<ProgramGroup, uint32_t> Pipeline::createCallablesProgram(const Context& ctx, const Module& module, const std::string& dc_name, const std::string& cc_name)
{
    m_callables_programs.emplace_back(ProgramGroup{});
    m_callables_programs.back().createCallables(ctx, module, dc_name, cc_name);
    return { m_callables_programs.back(), (uint32_t)(m_callables_programs.size() - 1) };
}

[[nodiscard]]
std::pair<ProgramGroup, uint32_t> Pipeline::createCallablesProgram(const Context& ctx, const ProgramEntry& dc_entry, const ProgramEntry& cc_entry)
{
    m_callables_programs.emplace_back(ProgramGroup{});
    m_callables_programs.back().createCallables(ctx, dc_entry, cc_entry);
    return { m_callables_programs.back(), (uint32_t)(m_callables_programs.size() - 1) };
}

[[nodiscard]]
ProgramGroup Pipeline::createExceptionProgram(const Context& ctx, const Module& module, const std::string& func_name)
{
    m_exception_program.createException(ctx, module, func_name);
    return m_exception_program;
}

[[nodiscard]]
ProgramGroup Pipeline::createExceptionProgram(const Context& ctx, const ProgramEntry& entry)
{
    m_exception_program.createException(ctx, entry);
    return m_exception_program;
}

// --------------------------------------------------------------------
void Pipeline::create(const Context& ctx)
{
    std::vector<OptixProgramGroup> optix_prg_groups;

    auto transform_programs = [&optix_prg_groups](const std::vector<ProgramGroup>& prg_groups)
    {
        std::transform(prg_groups.begin(), prg_groups.end(), std::back_inserter(optix_prg_groups),
            [](const ProgramGroup& prg) { return static_cast<OptixProgramGroup>(prg); });
    };

    optix_prg_groups.emplace_back(static_cast<OptixProgramGroup>(m_raygen_program));
    transform_programs(m_miss_programs);
    transform_programs(m_hitgroup_programs);
    transform_programs(m_callables_programs);
    if ((OptixProgramGroup)m_exception_program) optix_prg_groups.emplace_back(m_exception_program);

    // Create pipeline from program groups.
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixPipelineCreate(
        static_cast<OptixDeviceContext>(ctx),
        &m_compile_options,
        &m_link_options,
        optix_prg_groups.data(),
        static_cast<unsigned int>(optix_prg_groups.size()),
        log, 
        &sizeof_log, 
        &m_pipeline
    ));

    // Specify the max traversal depth and calculate the stack sizes.
    OptixStackSizes stack_sizes = {};
    for (auto& optix_prg_group : optix_prg_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(optix_prg_group, &stack_sizes));
    }
    
    uint32_t dc_stacksize_from_traversal;
    uint32_t dc_stacksize_from_state;
    uint32_t cc_stacksize;

    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        m_trace_depth,
        m_cc_depth,
        m_dc_depth,
        &dc_stacksize_from_traversal,
        &dc_stacksize_from_state,
        &cc_stacksize
    ));

    const uint32_t max_traversal_depth = 5; 
    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline,
        dc_stacksize_from_traversal, 
        dc_stacksize_from_state, 
        cc_stacksize, 
        max_traversal_depth
    ));
}

void Pipeline::destroy()
{
    if (m_pipeline) OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
    m_pipeline = nullptr;

    for (auto& module : m_modules)
        module.destroy();
    m_modules.clear();

    m_raygen_program.destroy();
    m_exception_program.destroy();

    auto destroyPrograms = [&](std::vector<ProgramGroup>& prgs)
    {
        for (auto& prg : prgs)
            prg.destroy();
        prgs.clear();
    };
    destroyPrograms(m_hitgroup_programs);
    destroyPrograms(m_miss_programs);
    destroyPrograms(m_callables_programs);
}

// --------------------------------------------------------------------
void Pipeline::setCompileOptions(const OptixPipelineCompileOptions& c_op)
{
    m_compile_options = c_op;
}

void Pipeline::enableMotionBlur()
{
    m_compile_options.usesMotionBlur = true;
}

void Pipeline::disableMotionBlur()
{
    m_compile_options.usesMotionBlur = false;
}

void Pipeline::setTraversableGraphFlags(const uint32_t flags)
{
    m_compile_options.traversableGraphFlags = flags;
}

void Pipeline::setNumPayloads(const int num_payloads)
{
    m_compile_options.numPayloadValues = num_payloads;
}

void Pipeline::setNumAttributes(const int num_attributes)
{
    m_compile_options.numAttributeValues = num_attributes;
}

void Pipeline::setLaunchVariableName(const char* params_name)
{
    m_compile_options.pipelineLaunchParamsVariableName = params_name;
}

void Pipeline::setExceptionFlags(OptixExceptionFlags flags)
{
    m_compile_options.exceptionFlags = flags;
}

OptixPipelineCompileOptions Pipeline::compileOptions() const
{
    return m_compile_options;
}

// --------------------------------------------------------------------
void Pipeline::setLinkOptions(const OptixPipelineLinkOptions& l_op)
{
    m_link_options = l_op;
}

void Pipeline::setLinkDebugLevel(const OptixCompileDebugLevel& debug_level)
{
    m_link_options.debugLevel = debug_level;
}

OptixPipelineLinkOptions Pipeline::linkOptions() const
{
    return m_link_options;
}

// --------------------------------------------------------------------
void Pipeline::setTraceDepth(const uint32_t depth)
{
    m_trace_depth = depth;
    m_link_options.maxTraceDepth = m_trace_depth;
}
uint32_t Pipeline::traceDepth() const
{
    return m_trace_depth;
}

// --------------------------------------------------------------------
void Pipeline::setContinuationCallableDepth(const uint32_t depth)
{
    m_cc_depth = depth;
}
uint32_t Pipeline::continuationCallableDepth() const
{
    return m_cc_depth;
}

// --------------------------------------------------------------------
void Pipeline::setDirectCallableDepth(const uint32_t depth)
{
    m_dc_depth = depth;
}
uint32_t Pipeline::directCallableDepth() const
{
    return m_dc_depth;
}


// Private functions -------------------------------------------------
void Pipeline::_initCompileOptions()
{
    m_compile_options.usesMotionBlur = false;
    m_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    m_compile_options.numPayloadValues = 2;
    m_compile_options.numAttributeValues = 3;
    m_compile_options.pipelineLaunchParamsVariableName = "";
#ifdef DEBUG
    m_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else   
    m_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
}

void Pipeline::_initLinkOptions()
{
    m_link_options.maxTraceDepth = 5;
    m_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
}

} // ::prayground