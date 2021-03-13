#pragma once

#include <optix.h>
#include <core/util.h>
#include <optix/program.h>

namespace pt {

class Pipeline {
public:
    explicit Pipeline(const std::string& params_name) {
        // Compile options;
        m_compile_options.usesMotionBlur = false;
        m_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_ALLOW_SINGLE_GAS;
        m_compile_options.numPayloadValues = 5;
        m_compile_options.numAttributeValues = 2;
        m_compile_options.pipelineLaunchParamsVariableName = params_name.c_str();
    #ifdef DEBUG
        m_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    #else   
        m_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    #endif

        // Link options;
        m_link_options.maxTraceDepth = 2;
        m_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

        // Depth settings
        _init_depth();
    }
    explicit Pipeline(OptixPipelineCompileOptions op) : m_compile_options(op) 
    { 
        _init_depth(); 
    }
    explicit Pipeline(OptixPipelineCompileOptions c_op, OptixPipelineLinkOptions l_op)
    : m_compile_options(c_op), m_link_options(l_op) 
    { 
        _init_depth(); 
    }

    explicit operator OptixPipeline() { return m_pipeline; }

    /** \brief Compile options. */
    void set_compile_options( const OptixPipelineCompileOptions& op ) { m_compile_options = op; }
    void use_motion_blur( bool is_use ) { m_compile_options.usesMotionBlur = is_use; }
    void set_traversable_graph_flags( unsigned int flags ) { m_compile_options.traversableGraphFlags = flags; }
    void set_num_payloads( const int num_payloads ) { m_compile_options.numPayloadValues = num_payload; }
    void set_num_attributes( const int num_attributes ) { m_compile_options.numAttributeValues = num_attributes; }
    void set_launch_variable_name( const std::string& params_name ) { m_compile_options.pipelineLaunchParamsVariableName = params_name.c_str(); }
    void set_exception_flags( const OptixExceptionFlags flags ) { m_compile_options.exceptionFlags = flags; }

    OptixPipelineCompileOptions compile_options() const { return m_compile_options; }

    /** \brief Link options */
    void set_link_options(const OptixPipelineLinkOptions& op) { m_link_options = op; }
    void set_link_trace_depth( const unsigned int depth ) { m_link_options.maxTraceDepth = depth; }
    void set_link_debug_level( const OptixCompileDebugLevel& debug_level ) { m_link_options.debugLevel = debug_level; }
    OptixPipelineLinkOptions link_options() const { return m_link_options; }

    /** \brief Create pipeline object and calculate the stack sizes of pipeline. */
    void create(const OptixDeviceContext &ctx, const std::vector<ProgramGroup>& prg_groups) {
        // Create pipeline from program groups.
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelinCreate(
            ctx,
            &m_compile_options,
            &m_link_options,
            prg_groups.data(),
            prg_groups.size(),
            log, 
            &sizeof_log, 
            m_pipeline
        ));

        // Specify the max traversal depth and calculate the stack sizes.
        OptixStackSizes stack_sizes = {};
        for (auto& prg_group : prg_groups) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes((OptixProgramGroup)prg_group, &stack_sizes));
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

        const uint32_t max_traversal_depth = 1; // 1 is specific or ordinary?
        OPTIX_CHECK(optixPipelineSetStackSize(
            m_pipeline,
            dc_stacksize_from_traversal, 
            dc_stacksize_from_state, 
            cc_stacksize, 
            max_traversal_depth
        ));
    }

    /** \brief Depth of traversal */
    void set_trace_depth(uint32_t depth) { m_trace_depth = depth; }
    uint32_t get_trace_depth() const { return m_trace_depth; }

    /** \brief Depth of continuation-callable */
    void set_cc_depth(uint32_t depth) { m_cc_depth = depth; }
    uint32_t get_cc_depth() const { return m_cc_depth; }

    /** \brief Depth of direct-callable */
    void set_dc_depth(uint32_t depth) { m_dc_depth = depth; }
    uint32_t get_dc_depth() const { return m_dc_depth; }

private:
    void _init_depth() { m_trace_depth = 5; m_cc_depth = 0; m_dc_depth = 0; }
    OptixPipelineCompileOptions m_compile_options;
    OptixPipelineLinkOptions m_link_options;
    OptixPipeline m_pipeline;
    uint32_t m_trace_depth, m_cc_depth, m_dc_depth;
};

}