#include "module.h"
#include "../core/file_util.h"

namespace oprt {

// ------------------------------------------------------------------
Module::Module()
{
    m_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    m_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
}

Module::Module(const OptixModuleCompileOptions& options)
: m_options(options)
{

}

// ------------------------------------------------------------------
void Module::create(const Context& ctx, const std::filesystem::path& ptx_path, OptixPipelineCompileOptions pipeline_options)
{
    auto filepath = findDataPath(ptx_path);
    Assert(filepath, "The CUDA file to create module of '" + ptx_path.string() + "' is not found.");

    char log[2048];
    size_t sizeof_log = sizeof(log);

    /** @todo Disable to use sutil::getPtxString() */
    const std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, filepath.value().string().c_str());
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        static_cast<OptixDeviceContext(ctx), 
        &m_options, 
        &pipeline_options, 
        ptx.c_str(), 
        ptx.size(), 
        log, 
        &sizeof_log, 
        &m_module
    ));
}

void Module::destroy()
{
    OPTIX_CHECK(optixModuleDestroy(m_module));
}

// ------------------------------------------------------------------
void Module::setOptimizationLevel(OptixCompileOptimizationLevel optLevel)
{
    m_options.optLevel = optLevel;
}

void Module::setDebugLevel(OptixCompileDebugLevel debugLevel)
{
    m_options.debugLevel = debugLevel;
}

// ------------------------------------------------------------------
void Module::setBoundValues(size_t offset_in_bytes, size_t size_in_bytes, void* bound_value_ptr, char* annotation)
{
    OptixModuleCompileBoundValueEntry* bound_values = new OptixModuleCompileBoundValueEntry();
    bound_values->pipelineParamOffsetInBytes = offset_in_bytes;
    bound_values->sizeInBytes = size_in_bytes;
    bound_values->boundValuePtr = bound_value_ptr;
    bound_values->annotation = annotation;
    m_options.boundValues = bound_values;
}

void Module::setBoundValues(OptixModuleCompileBoundValueEntry* bound_values)
{
    m_options.boundValues = bound_values;
}

void Module::setNumBounds(unsigned int num_bound)
{
    m_options.numBoundValues = num_bound;
}

OptixModuleCompileOptions Module::compileOptions() const 
{ 
    return m_options; 
}

} // ::oprt