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

/** 
 * \note These functions are used in few cases, 
 * especially when you don't want to write \c name directly. 
 */
inline const char* rg_func_str(std::string name) { return ("__raygen__" + name).c_str(); }
inline const char* is_func_str(std::string name) { return ("__intersection__" + name).c_str(); }
inline const char* ah_func_str(std::string name) { return ("__anyhit__" + name).c_str(); }
inline const char* ch_func_str(std::string name) { return ("__closesthit__" + name).c_str(); }
inline const char* ms_func_str(std::string name) { return ("__miss__" + name).c_str(); }
inline const char* ex_func_str(std::string name) { return ("__exception__" + name).c_str(); }
inline const char* dc_func_str(std::string name) { return ("__direct_callabel__" + name).c_str(); }
inline const char* cc_func_str(std::string name) { return ("__continuation_callable__" + name).c_str(); }