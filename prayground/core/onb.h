#pragma once

#include <prayground/math/vec.h>

namespace prayground {

    struct Onb {
        INLINE HOSTDEVICE Onb(const Vec3f& n) 
        {
            normal = n;

            if (n.y() > 0.9999999f) tangent = Vec3f(1.0f, 0.0f, 0.0f);
            else tangent = Vec3f(0.0f, 1.0f, 0.0f);

            tangent -= n * dot(tangent, n);
            tangent = normalize(tangent);
            bitangent = cross(tangent, normal);
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