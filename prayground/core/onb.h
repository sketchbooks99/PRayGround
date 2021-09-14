#pragma once

#include <prayground/math/vec_math.h>

namespace prayground {

struct Onb {
    INLINE HOSTDEVICE Onb(const float3& n) {
        normal = n;

        if (fabs(normal.x) > fabs(normal.z)) {
            bitangent.x = -normal.y;
            bitangent.y = normal.x;
            bitangent.z = 0;
        }
        else {
            bitangent.x = 0;
            bitangent.y = -normal.z;
            bitangent.z = normal.y;
        }

        bitangent = normalize(bitangent);
        tangent = cross( bitangent, normal );
    }

    INLINE HOSTDEVICE void inverseTransform(float3& p) const {
        p = p.x*bitangent + p.y*tangent + p.z*normal;
    }

    float3 bitangent;
    float3 tangent;
    float3 normal;
};

}