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

#pragma once

#include <optix.h>
#ifndef __CUDACC__
    #include <stdexcept>
    #include <prayground/core/stream_helpers.h>
#endif

#ifdef __CUDACC__
    #define CALLABLE_FUNC extern "C" __device__
    #define INLINE __forceinline__
    #define HOSTDEVICE __device__ __host__
    #define HOST __host__
    #define DEVICE __device__
    #define GLOBAL __global__
#else
    #define CALLABLE_FUNC
    #define INLINE inline
    #define HOSTDEVICE
    #define HOST 
    #define DEVICE 
    #define GLOBAL
#endif

#define RG_FUNC(name) __raygen__ ## name
#define IS_FUNC(name) __intersection__ ## name
#define AH_FUNC(name) __anyhit__ ## name
#define CH_FUNC(name) __closesthit__ ## name
#define MS_FUNC(name) __miss__ ## name
#define EX_FUNC(name) __exception__ ## name
#define DC_FUNC(name) __direct_callable__ ## name
#define CC_FUNC(name) __continuation_callable__ ## name

#define RG_FUNC_TEXT(name) "__raygen__" name
#define IS_FUNC_TEXT(name) "__intersection__" name
#define AH_FUNC_TEXT(name) "__anyhit__" name
#define CH_FUNC_TEXT(name) "__closesthit__" name
#define MS_FUNC_TEXT(name) "__miss__" name
#define EX_FUNC_TEXT(name) "__exception__" name
#define DC_FUNC_TEXT(name) "__direct_callable__" name
#define CC_FUNC_TEXT(name) "__continuation_callable__" name

#ifndef __CUDACC__

// OptiX error handles -------------------------------------------------------------
#define OPTIX_CHECK(call)                                                       \
    do                                                                          \
    {                                                                           \
        OptixResult res = call;                                                 \
        if (res != OPTIX_SUCCESS)                                               \
        {                                                                       \
            std::stringstream ss;                                               \
            ss << "ERROR: " << res << ", ";                                     \
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

#endif // __CUDACC__