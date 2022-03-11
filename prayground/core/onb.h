#pragma once

#include <prayground/math/vec.h>

namespace prayground {

struct Onb {
    INLINE HOSTDEVICE Onb(const Vec3f& n) 
    {
        normal = n;

        if (fabs(normal.x()) > fabs(normal.z()))
            bitangent = Vec3f(-normal.y(), normal.x(), 0.0f);
        else
            bitangent = Vec3f(0.0f, -normal.z(), normal.y());

        bitangent = normalize(bitangent);
        tangent = cross( bitangent, normal );
    }

    INLINE HOSTDEVICE void inverseTransform(Vec3f& p) const 
    {
        p = p.x() * tangent + p.y() * bitangent + p.z() * normal;
    }

    Vec3f tangent;
    Vec3f bitangent;
    Vec3f normal;
};

}