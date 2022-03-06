#pragma once

#include <prayground/math/vec_math.h>

namespace prayground {

struct Onb {
    INLINE HOSTDEVICE Onb(const Vec3f& n) {
        normal = n;

        if (fabs(normal[0]) > fabs(normal[2])) {
            bitangent[0] = -normal[1];
            bitangent[1] = normal[0];
            bitangent[2] = 0;
        }
        else {
            bitangent[0] = 0;
            bitangent[1] = -normal[2];
            bitangent[2] = normal[1];
        }

        bitangent = normalize(bitangent);
        tangent = cross( bitangent, normal );
    }

    INLINE HOSTDEVICE void inverseTransform(Vec3f& p) const {
        p = p[0] * bitangent + p[1] * tangent + p[2] * normal;
    }

    Vec3f bitangent;
    Vec3f tangent;
    Vec3f normal;
};

}