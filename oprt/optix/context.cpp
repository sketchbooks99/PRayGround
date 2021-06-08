#include "context.h"

namespace oprt {

static void contextLogCallback( unsigned int level, const char* tag, const char* msg, void* /* cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
            << msg << "\n";
}

void Context::create()
{
    CUDA_CHECK(cudaFree(0));
    CUcontext cu_ctx = 0;
    OPTIX_CHECK( optixDeviceContextCreate(cu_ctx, &m_options, &m_ctx) );
}

}