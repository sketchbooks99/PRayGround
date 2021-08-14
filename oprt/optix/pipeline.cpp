#include "pipeline.h"

namespace oprt {

// --------------------------------------------------------------------
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
void Pipeline::create(const Context& ctx, const std::vector<ProgramGroup>& prg_groups)
{
    std::vector<OptixProgramGroup> optix_prg_groups;
    std::transform(prg_groups.begin(), prg_groups.end(), std::back_inserter(optix_prg_groups),
        [](ProgramGroup pg){ return static_cast<OptixProgramGroup>(pg); });

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
    OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
}

// --------------------------------------------------------------------
void Pipeline::setCompileOptions(const OptixPipelineCompileOptions& c_op)
{
    m_compile_options = c_op;
}

void Pipeline::useMotionBlur(const bool is_use)
{
    m_compile_options.useMotionBlur = is_use;
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

void Pipeline::setLaunchVariableName(const std::string& params_name)
{
    m_compile_options.pipelineLaunchParamsVariableName = params_name.c_str();
}

void Pipeline::setExceptionFlags(const OptixExceptionFlags& flags)
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
    m_dc_depth = depth;
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

} // ::oprt