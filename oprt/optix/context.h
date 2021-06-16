#pragma once

#include <optix.h>
#include "../core/util.h"

namespace oprt {

static void contextLogCallback( unsigned int level, const char* tag, const char* msg, void* cbdata)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
            << msg << "\n";
}

static OptixDeviceContextOptions default_options = 
{
    &contextLogCallback,                     // logCallbackFunction
    nullptr,                                 // logCallbackData
    4,                                       // logCallbackLevel
    OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF // validationMode
};

class Context {
public:
    explicit Context() : Context(0, default_options) {}

    explicit Context(const OptixDeviceContextOptions& options)
    : Context(0, options) {}

    explicit Context(unsigned int device_id)
    : Context(device_id, default_options) {}

    explicit Context(unsigned int device_id, const OptixDeviceContextOptions& options)
    : m_device_id(device_id), m_options(options) {}

    explicit operator OptixDeviceContext() const { return m_ctx; }
    explicit operator OptixDeviceContext&() { return m_ctx; }

    void create();

    // Setter for context options
    void setOptions(const OptixDeviceContextOptions& options) { m_options = options; }
    void setLogCallbackFunction(OptixLogCallback callback_func)
    {
        m_options.logCallbackFunction = callback_func;
    }
    void setLogCallbackData(void* callback_data)
    {
        m_options.logCallbackData = callback_data;
    }
    void setLogCallbackLevel(int callback_level)
    {
        m_options.logCallbackLevel = callback_level;
    }
    void validationEnabled(bool is_valid)
    {
        m_options.validationMode = is_valid ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
                                            : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    }
    
    unsigned int deviceId() const { return m_device_id; }
private:
    unsigned int m_device_id { 0 };
    OptixDeviceContext m_ctx;
    OptixDeviceContextOptions m_options;
};

}