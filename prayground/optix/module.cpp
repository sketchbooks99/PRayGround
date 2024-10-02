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
#include <prayground_config.h>
#include <prayground/core/file_util.h>
#include <prayground/core/util.h>
#include <prayground/app/app_runner.h>
#include <prayground/optix/macros.h>
#include <map>

#define STRINGIFY( x ) STRINGIFY2( x )
#define STRINGIFY2( x ) #x

// NVRTC error handles
#define NVRTC_CHECK(call)                                                                                                                  \
    do                                                                                                                                     \
    {                                                                                                                                      \
        nvrtcResult code = call;                                                                                                           \
        if (code != NVRTC_SUCCESS)                                                                                                         \
            throw std::runtime_error( "ERROR: " __FILE__ "(" STRINGIFY(__LINE__) "): " + std::string( nvrtcGetErrorString( code ) ) );   \
    } while (0)

namespace prayground {

    namespace fs = std::filesystem;

    namespace {
        struct SourceCache
        {
            std::map<std::string, std::string*> map;
            ~SourceCache()
            {
                for (std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end(); ++it)
                    delete it->second;
            }
        };
        SourceCache g_source_cache;

        bool readSourceFile(std::string& str, const std::string& filename)
        {
            std::ifstream file(filename.c_str(), std::ios::binary);
            if (file.good()) {
                std::vector<uint8_t> buffer = std::vector<uint8_t>(std::istreambuf_iterator<char>(file), {});
                str.assign(buffer.begin(), buffer.end());
                return true;
            }
            return false;
        }

        std::string fileExtensionForLoading() {
            std::string extension;
#if PRAYGROUND_INPUT_GENERATE_PTX
            extension = ".ptx";
#endif
#if PRAYGROUND_INPUT_GENERATE_OPTIXIR
            extension = ".optixir";
#endif
            if (const char* ext = getenv("PRAYGROUND_INPUT_EXTENSION")) {
                extension = ext;
                if (extension.size() && extension[0] != '.')
                    extension = "." + extension;
            }
            return extension;
        }

        std::string inputFilePath(std::string filepath) {
            std::string extension = fileExtensionForLoading();
            std::string dir = pgGetDir(filepath).string();
            std::string filename = pgGetFilename(filepath).string();

            std::string path = dir;
            path += "/";
            path += pgGetAppName();
            path += "_generated_";
            path += filename;
            path += extension;

            if (auto ret_filepath = pgFindDataPath(path)) {
                return ret_filepath.value().string();
            }
            ASSERT(false, "The file '" + path + "' is not found.");
        }

        static void getInputDataFromFile(std::string& input_data, std::string filepath)
        {
            const std::string input_filepath = inputFilePath(filepath);

            // Try to open source file
            if (!readSourceFile(input_data, input_filepath)) {
                std::string err = "Couldn't open source file " + filepath;
                throw std::runtime_error(err.c_str());
            }
        }

    #if CUDA_NVRTC_ENABLED
        std::string g_nvrtcLog;

        void getPtxFromCuString(std::string& ptx, const char* cu_source, const char* name, const char** log_string)
        {
            nvrtcProgram prog = 0;
            NVRTC_CHECK(nvrtcCreateProgram(&prog, cu_source, name, 0, nullptr, nullptr));

            std::vector<const char*> options;

            std::string app_dir;
            if (pgAppDir().string() != "") {
                app_dir = std::string("-I") + pgAppDir().string();
                options.push_back(app_dir.c_str());
            }

            std::string file_dir;
            if (name)
            {
                file_dir = std::string("-I") + fs::path(name).parent_path().string();
                options.push_back( file_dir.c_str() );
            }

            // Collect include dirs
            std::vector<std::string> include_dirs;
            const char* abs_dirs[] = { PRAYGROUND_ABSOLUTE_INCLUDE_DIRS };
            const char* rel_dirs[] = { PRAYGROUND_RELATIVE_INCLUDE_DIRS };

            for (const char* dir : abs_dirs)
                include_dirs.push_back(std::string("-I") + dir);
            for (const char* dir : rel_dirs)
                include_dirs.push_back(std::string("-I") + dir);
            for (const std::string& dir : include_dirs)
                options.push_back(dir.c_str());

            // Collect NVRTC options
            const char* compiler_options[] = { CUDA_NVRTC_OPTIONS };
            std::copy(std::begin(compiler_options), std::end(compiler_options), std::back_inserter(options));

            // JIT compile CU to PTX
            const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int)options.size(), options.data());

            // Retrieve log output
            size_t log_size = 0;
            NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
            g_nvrtcLog.resize(log_size);
            if (log_size > 1)
            {
                NVRTC_CHECK(nvrtcGetProgramLog(prog, &g_nvrtcLog[0]));
                if (log_string)
                    *log_string = g_nvrtcLog.c_str();
            }
            if (compileRes != NVRTC_SUCCESS)
                throw std::runtime_error("NVRTC Compilation failed.\n" + g_nvrtcLog);

            // Retrieve PTX code
            size_t ptx_size = 0;
            NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
            ptx.resize(ptx_size);
            NVRTC_CHECK(nvrtcGetPTX(prog, &ptx[0]));

            // Cleanup
            NVRTC_CHECK(nvrtcDestroyProgram(&prog));
        }
    #endif // CUDA_NVRTC_ENABLED

    } // nonamed namespace

    // ------------------------------------------------------------------
    Module::Module()
    {
        m_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        m_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    #if OPTIX_VERSION < 70400
        m_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
    #else 
        m_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    #endif
    }

    Module::Module(const OptixModuleCompileOptions& options)
    : m_options(options)
    {

    }

    // ------------------------------------------------------------------
    void Module::createFromCudaFile(const Context& ctx, const fs::path& filename, OptixPipelineCompileOptions pipeline_options)
    {
//        auto filepath = pgFindDataPath(filename);
//        ASSERT(filepath, "The CUDA file to create module of '" + filename.string() + "' is not found.");
//
//        const char** log = nullptr;
//        std::string key = filepath.value().string();
//        std::map<std::string, std::string*>::iterator elem = g_source_cache.map.find(key);
//        
//        // Load cuda source from file
//        std::string cu_source = pgGetTextFromFile(filepath.value());
//        std::string* ptx;
//        if (elem == g_source_cache.map.end())
//        {
//#if CUDA_NVRTC_ENABLED
//            ptx = new std::string;
//            getPtxFromCuString(*ptx, cu_source.c_str(), filepath.value().string().c_str(), log);
//            g_source_cache.map[key] = ptx;
//#else
//            g_source_cache.map[key] = cu_source;
//#endif
//        }
//        else
//        {
//            ptx = elem->second;
//        }
//        createFromPtxSource(ctx, *ptx, pipeline_options);

        auto filepath = pgFindDataPath(filename);
        ASSERT(filepath, "The CUDA file to create module of '" + filename.string() + "' is not found.");

        const char** log = nullptr;
        std::string* input_data;
        std::string key = filepath.value().string();
        std::map<std::string, std::string*>::iterator elem = g_source_cache.map.find(key);

        if (elem == g_source_cache.map.end())
        {
            input_data = new std::string();
#if CUDA_NVRTC_ENABLED
            std::string cu_source = pgGetTextFromFile(filepath.value());
            getPtxFromCuString(input_data, cu_source.c_str(), filepath.value().string().c_str(), log);
#else
            getInputDataFromFile(*input_data, filepath.value().string());
#endif
            g_source_cache.map[key] = input_data;
        }
        else
        {
            input_data = elem->second;
        }
        std::cout << *input_data << std::endl;
        createFromPtxSource(ctx, *input_data, pipeline_options);
    }

    void Module::createFromCudaSource(const Context& ctx, const std::string& source, OptixPipelineCompileOptions pipeline_options)
    {
#if CUDA_NVRTC_ENABLED
        const char** log = nullptr;
        std::string* ptx = new std::string;
        getPtxFromCuString(*ptx, source.c_str(), "", log);
        createFromPtxSource(ctx, *ptx, pipeline_options);
#endif
    }

    void Module::createFromPtxFile(const Context& ctx, const fs::path& filename, OptixPipelineCompileOptions pipeline_options)
    {
        auto filepath = pgFindDataPath(filename);
        ASSERT(filepath, "The PTX file to create module of '" + filename.string() + "' is not found.");

        std::string key = filepath.value().string();
        std::map<std::string, std::string*>::iterator elem = g_source_cache.map.find(key);
        std::string* ptx = new std::string;

        if (elem == g_source_cache.map.end())
        {
            *ptx = pgGetTextFromFile(filepath.value());
            g_source_cache.map[key] = ptx;
        }
        else
        {
            ptx = elem->second;
        }

        createFromPtxSource(ctx, *ptx, pipeline_options);
    }

    void Module::createFromPtxSource(const Context& ctx, const std::string& source, OptixPipelineCompileOptions pipeline_options)
    {
        char log[2048];
        size_t sizeof_log = sizeof(log);

    #if OPTIX_VERSION < 70700
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
    #else
        OPTIX_CHECK_LOG(optixModuleCreate(
            static_cast<OptixDeviceContext>(ctx),
            &m_options,
            &pipeline_options,
            source.c_str(),
            source.size(),
            log,
            &sizeof_log,
            &m_module
        ));
    #endif
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

    const OptixModuleCompileOptions& Module::compileOptions() const 
    { 
        return m_options; 
    }

    OptixModuleCompileOptions& Module::compileOptions()
    {
        return m_options;
    }

} // namespace prayground