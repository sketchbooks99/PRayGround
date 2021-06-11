#pragma once

#include <optix.h>
#include <optix_stack_size.h>
#include "../core/util.h"
#include "program.h"

namespace oprt {

class Pipeline {
public:
    explicit Pipeline(const std::string& params_name) {
        _initCompileOptions();
        m_compile_options.pipelineLaunchParamsVariableName = params_name.c_str();
        _initLinkOptions();
    }
    
    explicit Pipeline(const OptixPipelineCompileOptions& op) : m_compile_options(op) 
    { 
        _initLinkOptions();
    }
    
    explicit Pipeline(const OptixPipelineCompileOptions& c_op, const OptixPipelineLinkOptions& l_op)
    : m_compile_options(c_op), m_link_options(l_op) { }

    void destroy() {
        OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
    }

    explicit operator OptixPipeline() const { return m_pipeline; }
    explicit operator OptixPipeline&() { return m_pipeline; }

    /** \brief Compile options. */
    void setCompileOptions( const OptixPipelineCompileOptions& op ) { m_compile_options = op; }
    void useMotionBlur( bool is_use ) { m_compile_options.usesMotionBlur = is_use; }
    void setTraversableGraphFlags( unsigned int flags ) { m_compile_options.traversableGraphFlags = flags; }
    void setNumPayloads( int num_payloads ) { m_compile_options.numPayloadValues = num_payloads; }
    void setNumAttributes( int num_attributes ) { m_compile_options.numAttributeValues = num_attributes; }
    void setLaunchVariableName( const std::string& params_name ) { m_compile_options.pipelineLaunchParamsVariableName = params_name.c_str(); }
    void setExceptionFlags( const OptixExceptionFlags& flags ) { m_compile_options.exceptionFlags = flags; }

    OptixPipelineCompileOptions compileOptions() const { return m_compile_options; }

    /** \brief Link options */
    void setLinkOptions( const OptixPipelineLinkOptions& op ) { m_link_options = op; }
    void setLinkTraceDepth( unsigned int depth ) { m_link_options.maxTraceDepth = depth; }
    void setLinkDebugLevel( const OptixCompileDebugLevel& debug_level ) { m_link_options.debugLevel = debug_level; }
    OptixPipelineLinkOptions link_options() const { return m_link_options; }

    /** \brief Create pipeline object and calculate the stack sizes of pipeline. */
    void create( const OptixDeviceContext& ctx, const std::vector<ProgramGroup>& prg_groups) {

        std::vector<OptixProgramGroup> optix_prg_groups;
        std::transform(prg_groups.begin(), prg_groups.end(), std::back_inserter(optix_prg_groups),
            [](OptixProgramGroup pg){ return static_cast<OptixProgramGroup>(pg); });

        // Create pipeline from program groups.
        char log[2048];
        size_t sizeof_log = sizeof(log);

        OPTIX_CHECK_LOG(optixPipelineCreate(
            ctx,
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

    /** \brief Depth of traversal */
    void setTraceDepth(uint32_t depth) { 
        m_trace_depth = depth;
        m_link_options.maxTraceDepth = m_trace_depth;
    }
    uint32_t traceDepth() const { return m_trace_depth; }

    /** \brief Depth of continuation-callable */
    void setContinuationCallableDepth(uint32_t depth) { m_cc_depth = depth; }
    uint32_t continuationCallableDepth() const { return m_cc_depth; }

    /** \brief Depth of direct-callable */
    void setDirectCallableDepth(uint32_t depth) { m_dc_depth = depth; }
    uint32_t directCallableDepth() const { return m_dc_depth; }

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