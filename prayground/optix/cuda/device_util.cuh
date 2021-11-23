#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <prayground/optix/helpers.h>
#include <prayground/optix/macros.h>
#include <prayground/math/util.h>

#ifdef __CUDACC__
using int8_t = char;
using int16_t = short;
using int32_t = int;
using int64_t = long long;
using uint8_t = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;
#endif

#define PG_MAX_NUM_ATTRIBUTES 8
#define PG_MAX_NUM_PAYLOADS 8

namespace prayground {

    template <uint32_t i>
    INLINE DEVICE uint32_t getAttribute()
    {
        static_assert(Base + 1 < PG_MAX_NUM_ATTRIBUTES, 
            "Index to get attribute exceeds the maximum number of attributes (" PG_MAX_NUM_ATTRIBUTES ")");
        if constexpr (i == 0)
            optixGetAttribute_0();
        if constexpr (i == 1)
            optixGetAttribute_1();
        if constexpr (i == 2)
            optixGetAttribute_2();
        if constexpr (i == 3)
            optixGetAttribute_3();
        if constexpr (i == 4)
            optixGetAttribute_4();
        if constexpr (i == 5)
            optixGetAttribute_5();
        if constexpr (i == 6)
            optixGetAttribute_6();
        if constexpr (i == 7)
            optixGetAttribute_7();
    }

    template <uint32_t i>
    INLINE DEVICE void setAttribute(uint32_t value)
    {
        static_assert(Base + 1 < PG_MAX_NUM_ATTRIBUTES, 
            "Index to set attribute exceeds the maximum number of attributes (" PG_MAX_NUM_ATTRIBUTES ")");
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
        static_assert(Base + 1 < PG_MAX_NUM_PAYLOADS, 
            "Index to get payload exceeds the maximum number of payloads (" PG_MAX_NUM_PAYLOADS ")");
        if constexpr (i == 0)
            optixGetPayload_0();
        if constexpr (i == 1)
            optixGetPayload_1();
        if constexpr (i == 2)
            optixGetPayload_2();
        if constexpr (i == 3)
            optixGetPayload_3();
        if constexpr (i == 4)
            optixGetPayload_4();
        if constexpr (i == 5)
            optixGetPayload_5();
        if constexpr (i == 6)
            optixGetPayload_6();
        if constexpr (i == 7)
            optixGetPayload_7();
    }

    template <uint32_t i>
    INLINE DEVICE void setPayload(uint32_t value)
    {
        static_assert(Base + 1 < PG_MAX_NUM_PAYLOADS, 
            "Index to set payload exceeds the maximum number of payloads (" PG_MAX_NUM_PATLOADS ")");
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
    INLINE DEVICE float2 getFloat2FromAttribute()
    {
        return make_float2(
            __int_as_float(getAttribute<Base + 0>()), 
            __int_as_float(getAttribute<Base + 1>())
        );
    }

    template <uint32_t Base>
    INLINE DEVICE float3 getFloat3FromAttribute()
    {
        return make_float3(
            __int_as_float(getAttribute<Base + 0>()), 
            __int_as_float(getAttribute<Base + 1>()), 
            __int_as_float(getAttribute<Base + 2>())
        );
    }

    template <uint32_t Base>
    INLINE DEVICE float4 getFloat4FromAttribute()
    {
        return make_float4(
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

    template <uint32_t... Payloads>
    INLINE DEVICE void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        unsigned int           ray_type,
        unsigned int           ray_count, 
        Payloads...            payloads
    )
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