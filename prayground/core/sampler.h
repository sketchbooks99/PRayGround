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

        static HOSTDEVICE Vec2f get2D(uint32_t& prev)
        {
            return Vec2f{rnd(prev), rnd(prev)};
        }

        static HOSTDEVICE Vec3f get3D(uint32_t& prev)
        {
            return Vec3f{rnd(prev), rnd(prev), rnd(prev)};
        }
    };

    class SobolSampler {
    public:
        static HOSTDEVICE float get1D(uint32_t& prev)
        {
            return rnd(prev);
        }

        static HOSTDEVICE Vec2f get2D(uint32_t& prev)
        {
            return Vec2f{rnd(prev), rnd(prev)};
        }

        static HOSTDEVICE Vec3f get3D(uint32_t& prev)
        {
            return Vec3f{rnd(prev), rnd(prev), rnd(prev)};
        }
    };

} // namespace prayground