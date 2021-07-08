#include "context.h"

namespace oprt {

static void contextLogCallback( unsigned int level, const char* tag, const char* msg, void* cbdata)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
            << msg << "\n";
}

Context::Context()
: m_device_id(0)
{
    m_options = 
    { 
        &contextLogCallback,                     // logCallbackFunction
        nullptr,                                 // logCallbackData
        4,                                       // logCallbackLevel
        OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF // validationMode
    };
}

Context::Context(unsigned int device_id)
: m_device_id(device_id)
{
    m_options = 
    { 
        &contextLogCallback,                     // logCallbackFunction
        nullptr,                                 // logCallbackData
        4,                                       // logCallbackLevel
        OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF // validationMode
    };
}

void Context::create()
{
    /// Verify if the \c device_id exceeds the detected number of GPU devices.
    int32_t num_device;
    CUDA_CHECK( cudaGetDeviceCount( &num_device ) );
    Assert( m_device_id < num_device, "oprt::Context::create(): device_id of oprt::Context exceeds the detected number of GPU devices.");

    // Set device with specified id.
    cudaDeviceProp prop;
    CUDA_CHECK( cudaGetDeviceProperties( &prop, m_device_id ));
    CUDA_CHECK( cudaSetDevice( m_device_id ) );

    // Create OptiX context.
    CUDA_CHECK(cudaFree(0));
    CUcontext cu_ctx = 0;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &m_options, &m_ctx ) );
}

}