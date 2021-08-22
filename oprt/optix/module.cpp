//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "module.h"
#include <nvrtc.h>
#include <sampleConfig.h>
#include <oprt/core/file_util.h>
#include <oprt/core/util.h>
#include <oprt/optix/macros.h>
#include <sutil/sutil.h>

namespace oprt {

namespace fs = std::filesystem;

// NVRTC error handles
#define NVRTC_CHECK(call)                                                                                                   \
    do                                                                                                                      \
    {                                                                                                                       \
        nvrtcResult code = func;                                                                                            \
        if (code != NVRTC_SUCCESS)                                                                                          \
            throw std::runtime_error( "ERROR: " __FILE__ "(" STRINGIFY( __LINE__ ) "): " + std::string( nvrtcGetErrorString( code ) ) ); \
    } while (0)
    
static void getPtxFromCuString(std::string& ptx, const char* source, const char* name, const char** log_string)
{

}

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
void Module::createFromCudaFile(const Context& ctx, const fs::path& filename, OptixPipelineCompileOptions pipeline_options)
{
#if !(CUDA_NVRTC_ENABLED)
    static_assert(false);
#endif

    auto filepath = findDataPath(filename);
    Assert(filepath, "oprt::Module::createFromModule(): The CUDA file to create module of '" + filename.string() + "' is not found.");

    // ***** from sutil *****
    // std::string ptx = sutil::getPtxString(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, filepath.value().string().c_str());
    // createFromPtxSource(ctx, ptx, pipeline_options);

    // ***** my code (todo) *****
    std::string source = getTextFromFile(filepath.value());
    createFromCudaSource(ctx, source, pipeline_options);
}

void Module::createFromCudaSource(const Context& ctx, const std::string& source, OptixPipelineCompileOptions pipeline_options)
{
#if !(CUDA_NVRTC_ENABLED)
    static_assert(false);
#endif
    nvrtcProgram prog = 0;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, source.c_str(), ))
}

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