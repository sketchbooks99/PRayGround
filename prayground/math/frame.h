#pragma once

#include <prayground/math/vec.h>

namespace prayground {

    class Frame {
    public:
        HOSTDEVICE Frame() : x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) {}
        HOSTDEVICE Frame(Vec3f x, Vec3f y, Vec3f z) : x(x), y(y), z(z) {}

        static HOSTDEVICE Frame FromXZ(Vec3f x, Vec3f z) {
            return Frame(x, cross(z, x), z);
        }

        static HOSTDEVICE Frame FromXY(Vec3f x, Vec3f y) {
            return Frame(x, y, cross(x, y));
        }

        Vec3f toLocal(Vec3f v) const {
            return Vec3f(dot(v, x), dot(v, y), dot(v, z));
        }
        
        Vec3f fromLocal(Vec3f v) const {
            return v.x() * x + v.y() * y + v.z() * z;
        }
    private:
        Vec3f x, y, z;
    };

} // namespace prayground