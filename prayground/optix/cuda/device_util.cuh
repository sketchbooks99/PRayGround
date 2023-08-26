#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <prayground/core/util.h>
#include <prayground/optix/util.h>
#include <prayground/math/util.h>
#include <prayground/math/vec.h>

#if OPTIX_VERSION >= 70400
#define PG_MAX_NUM_ATTRIBUTES 32
#define PG_MAX_NUM_PAYLOADS 32
#define PG_MAX_NUM_ATTRIBUTES_STR "32"
#define PG_MAX_NUM_PAYLOADS_STR "32"
#else
#define PG_MAX_NUM_ATTRIBUTES 8
#define PG_MAX_NUM_PAYLOADS 8
#define PG_MAX_NUM_ATTRIBUTES_STR "8"
#define PG_MAX_NUM_PAYLOADS_STR "8"
#endif

#ifdef __CUDACC__

#include <cuda/std/type_traits>

namespace prayground {

    INLINE DEVICE void* unpackPointer( uint32_t i0, uint32_t i1 )
    {
        const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
        void* ptr = reinterpret_cast<void*>( uptr );
        return ptr;
    }

    INLINE DEVICE void packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template <uint32_t i>
    INLINE DEVICE uint32_t getAttribute()
    {
        static_assert(i < PG_MAX_NUM_ATTRIBUTES, 
            "Index to get attribute exceeds the maximum number of attributes (" PG_MAX_NUM_ATTRIBUTES_STR ")");
        if constexpr (i == 0) return optixGetAttribute_0();
        if constexpr (i == 1) return optixGetAttribute_1();
        if constexpr (i == 2) return optixGetAttribute_2();
        if constexpr (i == 3) return optixGetAttribute_3();
        if constexpr (i == 4) return optixGetAttribute_4();
        if constexpr (i == 5) return optixGetAttribute_5();
        if constexpr (i == 6) return optixGetAttribute_6();
        if constexpr (i == 7) return optixGetAttribute_7();
#if OPTIX_VERSION >= 70400
        if constexpr (i == 8) return optixGetAttribute_8();
        if constexpr (i == 9) return optixGetAttribute_9();
        if constexpr (i == 10) return optixGetAttribute_10();
        if constexpr (i == 11) return optixGetAttribute_11();
        if constexpr (i == 12) return optixGetAttribute_12();
        if constexpr (i == 13) return optixGetAttribute_13();
        if constexpr (i == 14) return optixGetAttribute_14();
        if constexpr (i == 15) return optixGetAttribute_15();
        if constexpr (i == 16) return optixGetAttribute_16();
        if constexpr (i == 17) return optixGetAttribute_17();
        if constexpr (i == 18) return optixGetAttribute_18();
        if constexpr (i == 19) return optixGetAttribute_19();
        if constexpr (i == 20) return optixGetAttribute_20();
        if constexpr (i == 21) return optixGetAttribute_21();
        if constexpr (i == 22) return optixGetAttribute_22();
        if constexpr (i == 23) return optixGetAttribute_23();
        if constexpr (i == 24) return optixGetAttribute_24();
        if constexpr (i == 25) return optixGetAttribute_25();
        if constexpr (i == 26) return optixGetAttribute_26();
        if constexpr (i == 27) return optixGetAttribute_27();
        if constexpr (i == 28) return optixGetAttribute_28();
        if constexpr (i == 29) return optixGetAttribute_29();
        if constexpr (i == 30) return optixGetAttribute_30();
        if constexpr (i == 31) return optixGetAttribute_31();
#endif
    }

    template <uint32_t i>
    INLINE DEVICE uint32_t getPayload()
    {
        static_assert(i < PG_MAX_NUM_PAYLOADS, 
            "Index to get payload exceeds the maximum number of payloads (" PG_MAX_NUM_PAYLOADS_STR ")");
        if constexpr (i == 0) return optixGetPayload_0();
        if constexpr (i == 1) return optixGetPayload_1();
        if constexpr (i == 2) return optixGetPayload_2();
        if constexpr (i == 3) return optixGetPayload_3();
        if constexpr (i == 4) return optixGetPayload_4();
        if constexpr (i == 5) return optixGetPayload_5();
        if constexpr (i == 6) return optixGetPayload_6();
        if constexpr (i == 7) return optixGetPayload_7();
#if OPTIX_VERSION >= 70400
        if constexpr (i == 8) return optixGetPayload_8();
        if constexpr (i == 9) return optixGetPayload_9();
        if constexpr (i == 10) return optixGetPayload_10();
        if constexpr (i == 11) return optixGetPayload_11();
        if constexpr (i == 12) return optixGetPayload_12();
        if constexpr (i == 13) return optixGetPayload_13();
        if constexpr (i == 14) return optixGetPayload_14();
        if constexpr (i == 15) return optixGetPayload_15();
        if constexpr (i == 16) return optixGetPayload_16();
        if constexpr (i == 17) return optixGetPayload_17();
        if constexpr (i == 18) return optixGetPayload_18();
        if constexpr (i == 19) return optixGetPayload_19();
        if constexpr (i == 20) return optixGetPayload_20();
        if constexpr (i == 21) return optixGetPayload_21();
        if constexpr (i == 22) return optixGetPayload_22();
        if constexpr (i == 23) return optixGetPayload_23();
        if constexpr (i == 24) return optixGetPayload_24();
        if constexpr (i == 25) return optixGetPayload_25();
        if constexpr (i == 26) return optixGetPayload_26();
        if constexpr (i == 27) return optixGetPayload_27();
        if constexpr (i == 28) return optixGetPayload_28();
        if constexpr (i == 29) return optixGetPayload_29();
        if constexpr (i == 30) return optixGetPayload_30();
        if constexpr (i == 31) return optixGetPayload_31();
#endif
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

    template <uint32_t Base> 
    INLINE DEVICE Vec2f getVec2fFromPayload()
    {
        return Vec2f(
            __int_as_float(getPayload<Base + 0>()), 
            __int_as_float(getPayload<Base + 1>())
        );
    }

    template <uint32_t Base>
    INLINE DEVICE Vec3f getVec3fFromPayload()
    {
        return Vec3f(
            __int_as_float(getPayload<Base + 0>()), 
            __int_as_float(getPayload<Base + 1>()), 
            __int_as_float(getPayload<Base + 2>())
        );
    }

    template <uint32_t Base>
    INLINE DEVICE Vec4f getVec4fFromPayload()
    {
        return Vec4f(
            __int_as_float(getPayload<Base + 0>()),
            __int_as_float(getPayload<Base + 1>()),
            __int_as_float(getPayload<Base + 2>()),
            __int_as_float(getPayload<Base + 3>())
        );
    }

    template <typename ReturnT, uint32_t Base>
    INLINE DEVICE ReturnT* getPtrFromTwoAttributes()
    {
        const uint32_t u0 = getAttribute<Base + 0>();
        const uint32_t u1 = getAttribute<Base + 1>();
        return reinterpret_cast<ReturnT*>(unpackPointer(u0, u1));
    }

    template <typename ReturnT, uint32_t Base>
    INLINE DEVICE ReturnT* getPtrFromTwoPayloads()
    {
        const uint32_t u0 = getPayload<Base + 0>();
        const uint32_t u1 = getPayload<Base + 1>();
        return reinterpret_cast<ReturnT*>(unpackPointer(u0, u1));
    }

    template <uint32_t i>
    INLINE DEVICE void setAttribute(uint32_t attribute)
    {
        static_assert(i < PG_MAX_NUM_ATTRIBUTES,
            "Index to get attribute exceeds the maximum number of attributes (" PG_MAX_NUM_ATTRIBUTES_STR ")");
        if constexpr (i == 0) return optixSetAttribute_0();
        if constexpr (i == 1) return optixSetAttribute_1();
        if constexpr (i == 2) return optixSetAttribute_2();
        if constexpr (i == 3) return optixSetAttribute_3();
        if constexpr (i == 4) return optixSetAttribute_4();
        if constexpr (i == 5) return optixSetAttribute_5();
        if constexpr (i == 6) return optixSetAttribute_6();
        if constexpr (i == 7) return optixSetAttribute_7();
#if OPTIX_VERSION >= 70400
        if constexpr (i == 8) return optixSetAttribute_8();
        if constexpr (i == 9) return optixSetAttribute_9();
        if constexpr (i == 10) return optixSetAttribute_10();
        if constexpr (i == 11) return optixSetAttribute_11();
        if constexpr (i == 12) return optixSetAttribute_12();
        if constexpr (i == 13) return optixSetAttribute_13();
        if constexpr (i == 14) return optixSetAttribute_14();
        if constexpr (i == 15) return optixSetAttribute_15();
        if constexpr (i == 16) return optixSetAttribute_16();
        if constexpr (i == 17) return optixSetAttribute_17();
        if constexpr (i == 18) return optixSetAttribute_18();
        if constexpr (i == 19) return optixSetAttribute_19();
        if constexpr (i == 20) return optixSetAttribute_20();
        if constexpr (i == 21) return optixSetAttribute_21();
        if constexpr (i == 22) return optixSetAttribute_22();
        if constexpr (i == 23) return optixSetAttribute_23();
        if constexpr (i == 24) return optixSetAttribute_24();
        if constexpr (i == 25) return optixSetAttribute_25();
        if constexpr (i == 26) return optixSetAttribute_26();
        if constexpr (i == 27) return optixSetAttribute_27();
        if constexpr (i == 28) return optixSetAttribute_28();
        if constexpr (i == 29) return optixSetAttribute_29();
        if constexpr (i == 30) return optixSetAttribute_30();
        if constexpr (i == 31) return optixSetAttribute_31();
#endif
    }

    template <uint32_t i>
    INLINE DEVICE void setPayload(uint32_t payload)
    {
        static_assert(i < PG_MAX_NUM_PAYLOADS,
            "Index to get payload exceeds the maximum number of payloads (" PG_MAX_NUM_PAYLOADS_STR ")");
        if constexpr (i == 0) return optixSetPayload_0();
        if constexpr (i == 1) return optixSetPayload_1();
        if constexpr (i == 2) return optixSetPayload_2();
        if constexpr (i == 3) return optixSetPayload_3();
        if constexpr (i == 4) return optixSetPayload_4();
        if constexpr (i == 5) return optixSetPayload_5();
        if constexpr (i == 6) return optixSetPayload_6();
        if constexpr (i == 7) return optixSetPayload_7();
#if OPTIX_VERSION >= 70400
        if constexpr (i == 8) return optixSetPayload_8();
        if constexpr (i == 9) return optixSetPayload_9();
        if constexpr (i == 10) return optixSetPayload_10();
        if constexpr (i == 11) return optixSetPayload_11();
        if constexpr (i == 12) return optixSetPayload_12();
        if constexpr (i == 13) return optixSetPayload_13();
        if constexpr (i == 14) return optixSetPayload_14();
        if constexpr (i == 15) return optixSetPayload_15();
        if constexpr (i == 16) return optixSetPayload_16();
        if constexpr (i == 17) return optixSetPayload_17();
        if constexpr (i == 18) return optixSetPayload_18();
        if constexpr (i == 19) return optixSetPayload_19();
        if constexpr (i == 20) return optixSetPayload_20();
        if constexpr (i == 21) return optixSetPayload_21();
        if constexpr (i == 22) return optixSetPayload_22();
        if constexpr (i == 23) return optixSetPayload_23();
        if constexpr (i == 24) return optixSetPayload_24();
        if constexpr (i == 25) return optixSetPayload_25();
        if constexpr (i == 26) return optixSetPayload_26();
        if constexpr (i == 27) return optixSetPayload_27();
        if constexpr (i == 28) return optixSetPayload_28();
        if constexpr (i == 29) return optixSetPayload_29();
        if constexpr (i == 30) return optixSetPayload_30();
        if constexpr (i == 31) return optixSetPayload_31();
#endif
    }

    template <typename T>
    INLINE DEVICE void swap(T& a, T& b)
    {
        T c(a); a = b; b = c;
    }

} // namespace prayground

#endif
