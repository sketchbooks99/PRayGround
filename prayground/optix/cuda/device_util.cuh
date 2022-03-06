#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <prayground/optix/helpers.h>
#include <prayground/optix/macros.h>
#include <prayground/math/util.h>
#include <prayground/math/vec.h>

#ifdef __CUDACC__
#if !defined(int8_t)
    typedef signed char int8_t;
#endif

#if !defined(int16_t)
    typedef short int16_t;
#endif

#if !defined(int32_t)
    typedef int int32_t;
#endif

#if !defined(int64_t)
    typedef long long int64_t;
#endif

#if !defined(uint8_t)
    typedef unsigned char uint8_t;
#endif

#if !defined(uint16_t)
    typedef unsigned short uint16_t;
#endif

#if !defined(uint32_t)
    typedef unsigned int uint32_t;
#endif 

#if !defined(uint64_t)
    typedef unsigned long long uint64_t;
#endif

#endif // __CUDACC__

#define PG_MAX_NUM_ATTRIBUTES 8
#define PG_MAX_NUM_PAYLOADS 8
#define PG_MAX_NUM_ATTRIBUTES_STR "8"
#define PG_MAX_NUM_PAYLOADS_STR "8"

#ifdef __CUDACC__

namespace prayground {

    template <uint32_t i>
    INLINE DEVICE uint32_t getAttribute()
    {
        static_assert(i < PG_MAX_NUM_ATTRIBUTES, 
            "Index to get attribute exceeds the maximum number of attributes (" PG_MAX_NUM_ATTRIBUTES_STR ")");
        if constexpr (i == 0)
            return optixGetAttribute_0();
        if constexpr (i == 1)
            return optixGetAttribute_1();
        if constexpr (i == 2)
            return optixGetAttribute_2();
        if constexpr (i == 3)
            return optixGetAttribute_3();
        if constexpr (i == 4)
            return optixGetAttribute_4();
        if constexpr (i == 5)
            return optixGetAttribute_5();
        if constexpr (i == 6)
            return optixGetAttribute_6();
        if constexpr (i == 7)
            return optixGetAttribute_7();
    }

    template <uint32_t i>
    INLINE DEVICE void setAttribute(uint32_t value)
    {
        static_assert(i < PG_MAX_NUM_ATTRIBUTES, 
            "Index to set attribute exceeds the maximum number of attributes (" PG_MAX_NUM_ATTRIBUTES_STR ")");
        if constexpr (i == 0)
            optixSetAttribute_0(value);
        if constexpr (i == 1)
            optixSetAttribute_1(value);
        if constexpr (i == 2)
            optixSetAttribute_2(value);
        if constexpr (i == 3)
            optixSetAttribute_3(value);
        if constexpr (i == 4)
            optixSetAttribute_4(value);
        if constexpr (i == 5)
            optixSetAttribute_5(value);
        if constexpr (i == 6)
            optixSetAttribute_6(value);
        if constexpr (i == 7)
            optixSetAttribute_7(value);
    }

    template <uint32_t i>
    INLINE DEVICE uint32_t getPayload()
    {
        static_assert(i < PG_MAX_NUM_PAYLOADS, 
            "Index to set attribute exceeds the maximum number of payloads (" PG_MAX_NUM_PAYLOADS_STR ")");
        if constexpr (i == 0)
            return optixGetPayload_0();
        if constexpr (i == 1)
            return optixGetPayload_1();
        if constexpr (i == 2)
            return optixGetPayload_2();
        if constexpr (i == 3)
            return optixGetPayload_3();
        if constexpr (i == 4)
            return optixGetPayload_4();
        if constexpr (i == 5)
            return optixGetPayload_5();
        if constexpr (i == 6)
            return optixGetPayload_6();
        if constexpr (i == 7)
            return optixGetPayload_7();
    }

    template <uint32_t i>
    INLINE DEVICE void setPayload(uint32_t value)
    {
        static_assert(i < PG_MAX_NUM_PAYLOADS, 
            "Index to set payload exceeds the maximum number of payloads (" PG_MAX_NUM_PAYLOADS_STR ")");
        if constexpr (i == 0)
            optixSetPayload_0(value);
        if constexpr (i == 1)
            optixSetPayload_1(value);
        if constexpr (i == 2)
            optixSetPayload_2(value);
        if constexpr (i == 3)
            optixSetPayload_3(value);
        if constexpr (i == 4)
            optixSetPayload_4(value);
        if constexpr (i == 5)
            optixSetPayload_5(value);
        if constexpr (i == 6)
            optixSetPayload_6(value);
        if constexpr (i == 7)
            optixSetPayload_7(value);
    }

    template <uint32_t Base> 
    INLINE DEVICE Vec2f getVec2fFromAttribute()
    {
        return Vec2f(
            __int_as_float(getAttribute<Base + 0>()), 
            __int_as_float(getAttribute<Base + 1>())
        );
    }

    template <uint32_t Base>
    INLINE DEVICE Vec3f getVec3fFromAttribute()
    {
        return Vec3f(
            __int_as_float(getAttribute<Base + 0>()), 
            __int_as_float(getAttribute<Base + 1>()), 
            __int_as_float(getAttribute<Base + 2>())
        );
    }

    template <uint32_t Base>
    INLINE DEVICE Vec4f getVec4fFromAttribute()
    {
        return Vec4f(
            __int_as_float(getAttribute<Base + 0>()),
            __int_as_float(getAttribute<Base + 1>()),
            __int_as_float(getAttribute<Base + 2>()),
            __int_as_float(getAttribute<Base + 3>())
        );
    }

    template <typename T>
    INLINE DEVICE void swap(T& a, T& b)
    {
        T c(a); a = b; b = c;
    }

    INLINE DEVICE void* unpackPointer( unsigned int i0, unsigned int i1 )
    {
        const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
        void* ptr = reinterpret_cast<void*>( uptr );
        return ptr;
    }

    INLINE DEVICE void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
    {
        const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template <typename... Payloads>
    INLINE DEVICE auto trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        uint32_t               ray_type,
        uint32_t               ray_count, 
        Payloads...            payloads
    ) 
        -> decltype(std::initializer_list<uint32_t>{payloads...}, void())
    {
        optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            ray_type,        
            ray_count,           
            ray_type, 
            payloads...);
    }

} // ::prayground

#endif // __CUDACC__