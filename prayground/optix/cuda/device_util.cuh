#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <prayground/core/util.h>
#include <prayground/optix/util.h>
#include <prayground/math/util.h>
#include <prayground/math/vec.h>

#define PG_MAX_NUM_ATTRIBUTES 8
#define PG_MAX_NUM_PAYLOADS 8
#define PG_MAX_NUM_ATTRIBUTES_STR "8"
#define PG_MAX_NUM_PAYLOADS_STR "8"

#ifdef __CUDACC__

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
    INLINE DEVICE uint32_t getPayload()
    {
        static_assert(i < PG_MAX_NUM_PAYLOADS, 
            "Index to get payload exceeds the maximum number of payloads (" PG_MAX_NUM_PAYLOADS_STR ")");
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

    template <typename T>
    INLINE DEVICE void swap(T& a, T& b)
    {
        T c(a); a = b; b = c;
    }

} // namespace prayground

#endif
