#pragma once

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

// MACROs to easily define the function.
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
#include <string>
/** 
 * \note These functions are used in few cases, 
 * especially when you don't want to write \c name directly. 
 */
inline std::string rg_func_str(const std::string& name) { return "__raygen__" + name; }
inline std::string is_func_str(const std::string& name) { return "__intersection__" + name; }
inline std::string ah_func_str(const std::string& name) { return "__anyhit__" + name; }
inline std::string ch_func_str(const std::string& name) { return "__closesthit__" + name; }
inline std::string ms_func_str(const std::string& name) { return "__miss__" + name; }
inline std::string ex_func_str(const std::string& name) { return "__exception__" + name; }
inline std::string dc_func_str(const std::string& name) { return "__direct_callable__" + name; }
inline std::string cc_func_str(const std::string& name) { return "__continuation_callable__" + name; }

#endif