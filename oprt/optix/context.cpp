#include "context.h"

namespace oprt {

void Context::create()
{
    CUDA_CHECK(cudaFree(0));
    CUcontext cu_ctx = 0;
    OPTIX_CHECK( optixDeviceContextCreate(cu_ctx, &m_options, &m_ctx) );
}

}