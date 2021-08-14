#pragma once

#include <optix.h>
#include <optix_stack_size.h>
#include "../core/util.h"
#include "program.h"
#include "context.h"

namespace oprt {

class Pipeline {
public:
    explicit Pipeline(const std::string& params_name);
    explicit Pipeline(const OptixPipelineCompileOptions& op);
    explicit Pipeline(const OptixPipelineCompileOptions& c_op, const OptixPipelineLinkOptions& l_op);

    explicit operator OptixPipeline() const { return m_pipeline; }
    explicit operator OptixPipeline&() { return m_pipeline; }

    /** Create pipeline object and calculate the stack sizes of pipeline. */
    void create( const Context& ctx, const std::vector<ProgramGroup>& prg_groups);
    void destroy();

    /** Compile options. */
    void setCompileOptions( const OptixPipelineCompileOptions& op );
    void useMotionBlur(const bool is_use);
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
    void _initCompileOptions() { 
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
    void _initLinkOptions() {
        m_link_options.maxTraceDepth = 5;
        m_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    }
    OptixPipelineCompileOptions m_compile_options = {};
    OptixPipelineLinkOptions m_link_options = {};
    OptixPipeline m_pipeline { nullptr };
    uint32_t m_trace_depth { 5 }; 
    uint32_t m_cc_depth { 0 }; 
    uint32_t m_dc_depth { 0 };
};

}