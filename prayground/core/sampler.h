#pragma once

#include <prayground/optix/macros.h>
#include <prayground/math/vec.h>
#include <prayground/math/random.h>

namespace prayground {

    class UniformSampler {
    public:
        static HOSTDEVICE float get1D(uint32_t& prev)
        {
            return rnd(prev);
        }

        template <typename V2>
        static HOSTDEVICE V2 get2D(uint32_t& prev)
        {
            if constexpr (std::is_same_v<V2, float2>())
                return make_float2(rnd(prev), rnd(prev));
            else if constexpr (std::is_same_v<V2, Vec2f>())
                return Vec2f{rnd(prev), rnd(prev)};
            else 
                static_assert(false);
        }

        template <typename V3>
        static HOSTDEVICE V3 get3D(uint32_t& prev)
        {
            if constexpr (std::is_same_v<V3, float3>())
                return make_float3(rnd(prev), rnd(prev), rnd(prev));
            else if constexpr (std::is_same_v<V3, Vec3f>())
                return Vec3f{rnd(prev), rnd(prev), rnd(prev)};
            else
                static_assert(false);
        }
    };

    class SobolSampler {
    public:
        static HOSTDEVICE float get1D(uint32_t& prev)
        {
            return rnd(prev);
        }

        template <typename V2>
        static HOSTDEVICE V2 get2D(uint32_t& prev)
        {
            if constexpr (std::is_same_v<V2, float2>())
                return make_float2(rnd(prev), rnd(prev));
            else if constexpr (std::is_same_v<V2, Vec2f>())
                return Vec2f{rnd(prev), rnd(prev)};
        }

        template <typename V3>
        static HOSTDEVICE V3 get3D(uint32_t& prev)
        {
            if constexpr (std::is_same_v<V3, float3>())
                return make_float3(rnd(prev), rnd(prev), rnd(prev));
            else if constexpr (std::is_same_v<V3, Vec3f>())
                return Vec3f{rnd(prev), rnd(prev), rnd(prev)};
        }
    private:
        
    };

} // namespace prayground