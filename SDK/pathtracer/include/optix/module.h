#pragma once 

#include <optix.h>
#include "core_util.h"

namespace pt {

class Module {
public:
    explicit Module() {
        m_ptx_path = "";
        m_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        m_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        m_options.debugLevel = OPTIX_COMPILE_DEBUG_LINEINFO;
    }
    explicit Module(const std::string& ptx_path) : m_ptx_path(ptx_path) {
        m_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        m_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        m_options.debugLevel = OPTIX_COMPILE_DEBUG_LINEINFO;
    }
    explicit Module(const std::string& ptx_path, const OptixModuleCompileOptions& options)
    : m_ptx_path(ptx_path), m_options(options) {}

    explicit operator OptixModule() { return m_module; }
    
    void setpath(const std::string& ptx_path) { 
        m_ptx_path = ptx_path;
    } 
    void create(const OptixDeviceContext& context) {
        Assert(m_ptx_path != "", "Please configure the ptx module path.");
    }
private:
    OptixModule m_module;
    OptixModuleCompileOptions m_options;
    std::string m_ptx_path;
}; 

}