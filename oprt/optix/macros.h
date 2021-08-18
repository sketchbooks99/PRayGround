#pragma once

#include <optix.h>
#include <stdexcept>

#ifdef __CUDACC__
    #define CALLABLE_FUNC extern "C" __device__
    #define INLINE __forceinline__
    #define HOSTDEVICE __device__ __host__
    #define HOST __host__
    #define DEVICE __device__
#else
    #define CALLABLE_FUNC
    #define INLINE inline
    #define HOSTDEVICE
    #define HOST 
    #define DEVICE 
#endif

#define RG_FUNC(name) __raygen__ ## name
#define IS_FUNC(name) __intersection__ ## name
#define AH_FUNC(name) __anyhit__ ## name
#define CH_FUNC(name) __closesthit__ ## name
#define MS_FUNC(name) __miss__ ## name
#define EX_FUNC(name) __exception__ ## name
#define DC_FUNC(name) __direct_callable__ ## name
#define CC_FUNC(name) __continuation_callable__ ## name

#define RG_FUNC_STR(name) "__raygen__" name
#define IS_FUNC_STR(name) "__intersection__" name
#define AH_FUNC_STR(name) "__anyhit__" name
#define CH_FUNC_STR(name) "__closesthit__" name
#define MS_FUNC_STR(name) "__miss__" name
#define EX_FUNC_STR(name) "__exception__" name
#define DC_FUNC_STR(name) "__direct_callable__" name
#define CC_FUNC_STR(name) "__continuation_callable__" name

#ifndef __CUDACC__

// OptiX error handles -------------------------------------------------------------
#define OPTIX_CHECK(call)                                                       \
    do                                                                          \
    {                                                                           \
        OptixResult res = call;                                                 \
        if (res != OPTIX_SUCCESS)                                               \
        {                                                                       \
            std::stringstream ss;                                               \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"          \
               << __LINE__ << ")\n";                                            \
               throw std::runtime_error(ss.str());                              \
            }                                                                   \
        } while (0)

#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        const size_t sizeof_log_returned = sizeof_log;                         \
        sizeof_log = sizeof( log ); /* reset sizeof_log for future calls */    \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n" << log                               \
               << ( sizeof_log_returned > sizeof( log ) ? "<TRUNCATED>" : "" ) \
               << "\n";                                                        \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while( 0 )

// CUDA error handles --------------------------------------------------------------
#define CUDA_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while( 0 )


#define CUDA_SYNC_CHECK()                                                      \
    do                                                                         \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA error on synchronize with error '"                     \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw std::runtime_error(ss.str());                                \
        }                                                                      \
    } while( 0 )

#endif