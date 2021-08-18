#include "module.h"
#include <nvrtc.h>
#include <oprt/core/file_util.h>
#include <oprt/core/util.h>
#include <oprt/optix/macros.h>

namespace oprt {

namespace fs = std::filesystem;

// NVRTC error handles
#define NVRTC_CHECK(call)

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

 #ifdef CUDA_NVRTC_ENABLED
// ------------------------------------------------------------------
void Module::createFromCudaFile(const Context& ctx, const fs::path& filename, OptixPipelineCompileOptions pipeline_options)
{
    auto filepath = findDataPath(filename);
    Assert(filepath, "oprt::Module::createFromModule(): The CUDA file to create module of '" + filename.string() + "' is not found.");

    char log[2048];
    size_t sizeof_log = sizeof(log);

    std::string source = getTextFromFile(filepath.value());
}

void Module::createFromCudaSource(const Context& ctx, const std::string& source, OptixPipelineCompileOptions pipeline_options)
{
    nvrtcProgram prog = 0;
}
#endif

void Module::createFromPtxFile(const Context& ctx, const fs::path& filename, OptixPipelineCompileOptions pipeline_options)
{
    TODO_MESSAGE();
}

void Module::createFromPtxSource(const Context& ctx, const std::string& source, OptixPipelineCompileOptions pipeline_options)
{
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        static_cast<OptixDeviceContext>(ctx),
        &m_options,
        &pipeline_options,
        source.c_str(),
        source.size(),
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
void Module::setBoundValues(size_t offset_in_bytes, size_t size_in_bytes, void* bound_value_ptr, const char* annotation)
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