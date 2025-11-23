#pragma once

#include <prayground/math/vec.h>

namespace prayground {

    struct Onb {
        INLINE HOSTDEVICE Onb(const Vec3f& n) 
        {
            normal = n;

            Vec3f a = (fabs(normal[0]) > 0.999f) ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0);

            tangent = normalize(cross(normal, a));
            bitangent = cross(normal, tangent);
        }

        INLINE HOSTDEVICE void inverseTransform(Vec3f& p) const 
        {
            p = p[0] * bitangent + p[1] * tangent + p[2] * normal;
        }

        Vec3f bitangent;
        Vec3f tangent;
        Vec3f normal;
    };

}