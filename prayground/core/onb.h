#pragma once

#include <prayground/math/vec.h>

namespace prayground {

    struct Onb {
        INLINE HOSTDEVICE Onb(const Vec3f& n) 
        {
            normal = n;

            if (n.x() > 0.9f) bitangent = Vec3f(0.0f, 1.0f, 0.0f);
            else bitangent = Vec3f(1.0f, 0.0f, 0.0f);

            bitangent -= n * dot(bitangent, n);
            bitangent = normalize(bitangent);
            tangent = cross(bitangent, normal);
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