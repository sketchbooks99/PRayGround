#pragma once 

#include <optix.h>
#include <include/core/util.h>

namespace pt {

/**
 * \brief Module object to easily manage OptixModule and its options. 
 * 
 * \note The compile options are set to specified value at default. 
 * If you need to update them to appropriate values for your applications, 
 * please use copy constructor and setter effectively. 
 */
class Module {
public:
    explicit Module() {
        m_ptx_path = "";
        m_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        m_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        m_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    }
    explicit Module(const std::string& ptx_path) : m_ptx_path(ptx_path) {
        m_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        m_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        m_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    }
    explicit Module(const std::string& ptx_path, const OptixModuleCompileOptions& options)
    : m_ptx_path(ptx_path), m_options(options) {}

    explicit operator OptixModule() { return m_module; }

    void destroy() {
        OPTIX_CHECK(optixModuleDestroy(m_module));
    }
    
    void create(const OptixDeviceContext& ctx, const OptixPipelineCompileOptions& pipeline_options) {
        Assert(!m_ptx_path.empty(), "Please configure the ptx module path.");
        
        char log[2048];
        size_t sizeof_log = sizeof(log);
    
        const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, m_ptx_path.c_str());
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            ctx,
            &m_options,
            &pipeline_options, 
            ptx.c_str(),
            ptx.size(), 
            log,
            &sizeof_log,
            &m_module
        ));
    }

    void set_path(const std::string& ptx_path) { m_ptx_path = ptx_path; } 
    std::string get_path() const { return m_ptx_path; }

    /** \note At default, This is set to OPTIX_COMPILE_OPTIMIZATION_DEFAULT */
    void set_optlevel(const OptixCompileOptimizationLevel optlevel) { m_options.optLevel = optlevel; }
    /** \note At default, This is set to OPTIX_COMPILE_DEBUG_LINEINFO */
    void set_debuglevel(const OptixCompileDebugLevel debuglevel) { m_options.debugLevel = debuglevel; }

    /** \brief For specifying specializations for pipelineParams as specified in 
     *  OptixPipelineCompileOptions::pipelineLaunchParamsVariableName 
     * 
     *  \note Bound values are ignored if numBoundValues is set to 0, 
     *  and numBoundValues is 0 at default. */
    void set_boundvalues(const size_t offset_in_bytes, const size_t size_in_bytes, 
                         const void* bound_value_ptr, const char* annotation)
    {
        OptixModuleCompileBoundValueEntry* bound_values = new OptixModuleCompileBoundValueEntry();
        bound_values->pipelineParamOffsetInBytes = offset_in_bytes;
        bound_values->sizeInBytes = size_in_bytes;
        bound_values->boundValuePtr = bound_value_ptr;
        bound_values->annotation = annotation;
        m_options.boundValues = bound_values;
    }
    void set_boundvalues(const OptixModuleCompileBoundValueEntry* bound_values) { m_options.boundValues = bound_values; }
    void set_numbounds(const unsigned int num_bound) { m_options.numBoundValues = num_bound; }

    OptixModuleCompileOptions compile_options() const { return m_options; }

private:
    OptixModule m_module;
    std::string m_ptx_path;
    OptixModuleCompileOptions m_options {};
}; 

}