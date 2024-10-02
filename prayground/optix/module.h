#pragma once 

#include <optix.h>
#include <filesystem>
#include "context.h"

namespace prayground {

/**
 * @brief 
 * Module object to easily manage OptixModule and its options. 
 * 
 * @note 
 * The compile options are set to specified value at default. 
 * If you need to update them to appropriate values for your applications, 
 * please use copy constructor and setter effectively. 
 */
class Module {
public:
    explicit Module();
    explicit Module(const OptixModuleCompileOptions& options);

    explicit operator OptixModule() const { return m_module; }
    explicit operator OptixModule&() { return m_module; }
    void createFromCudaFile(const Context& ctx, const std::filesystem::path& filename, OptixPipelineCompileOptions pipeline_options);
    void createFromCudaSource(const Context& ctx, const std::string& source, OptixPipelineCompileOptions pipeline_options);
    void createFromPtxFile(const Context& ctx, const std::filesystem::path& filename, OptixPipelineCompileOptions pipeline_options);
    void createFromPtxSource(const Context& ctx, const std::string& source, OptixPipelineCompileOptions pipeline_options);

    void destroy();

    /** @note At default, This is set to OPTIX_COMPILE_OPTIMIZATION_DEFAULT */
    void setOptimizationLevel(OptixCompileOptimizationLevel optlevel);
    /** @note At default, This is set to OPTIX_COMPILE_DEBUG_LINEINFO */
    void setDebugLevel(OptixCompileDebugLevel debuglevel);

    /** 
     * @brief 
     * For specifying specializations for pipelineParams as specified in 
     * OptixPipelineCompileOptions::pipelineLaunchParamsVariableName 
     * 
     * @note 
     * Bound values are ignored if numBoundValues is set to 0, 
     * and numBoundValues is 0 at default. 
     */
    void setBoundValues( size_t offset_in_bytes, size_t size_in_bytes, void* bound_value_ptr, const char* annotation);
    void setBoundValues( OptixModuleCompileBoundValueEntry* bound_values);
    void setNumBounds( unsigned int num_bound );

    const OptixModuleCompileOptions& compileOptions() const;
    OptixModuleCompileOptions& compileOptions();

private:
    OptixModule m_module { nullptr };
    OptixModuleCompileOptions m_options {};
}; 

}