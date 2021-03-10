#pragma once 

#if !defined(__CUDACC__)
#include <string>
#include <stdexcept>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>
#include "stream_helpers.h"
#endif

#define HOST __host__
#define DEVICE __device__

#if !defined(__CUDACC__)
#define CALLABLE_FUNC
#define INLINE inline
#define HOSTDEVICE
#else 
#define CALLABLE_FUNC extern "C" __device__
#define INLINE __forceinline__
#define HOSTDEVICE __device__ __host__
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

namespace pt {

template <typename T>
void swap(T& a, T& b)
{
#ifdef __CUDACC__
    T c(a); a = b; b = c;
#else   
    std::swap(a, b);
#endif
}

float random_float() {
    return rand() / (RAND_MAX + 1.0);
}

float random_float(unsigned int seed) {
    std::mt19937 rnd_src(seed);
    std::uniform_real_distribution<float> rnd_dist(0,1);
    return rnd_dist(rnd_src);
}

#if !defined(__CUDACC__)
inline void Throw(const std::string& msg) {
    throw std::runtime_error(msg);
}

inline void Assert(bool condition, const std::string& msg) {
    if(!condition) Throw(msg);
}
#endif

}
