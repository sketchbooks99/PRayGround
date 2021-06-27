#pragma once

#include "../core/util.h"

namespace oprt {

struct Onb {
    INLINE HOSTDEVICE Onb(const float3& normal) {
        // m_normal = normal;

        // if (fabs(m_normal.x) > fabs(m_normal.z)) {
        //     m_binormal.x = -m_normal.y;
        //     m_binormal.y = m_normal.x;
        //     m_binormal.z = 0;
        // }
        // else {
        //     m_binormal.x = 0;
        //     m_binormal.y = -m_normal.z;
        //     m_binormal.z = m_normal.y;
        // }

        // m_binormal = normalize(m_binormal);
        // m_tangent = cross( m_binormal, m_normal );

        m_normal = normalize(normal);
        float3 a = (fabs(m_normal.x) > 0.9) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
        m_tangent = normalize(cross(m_normal, a));
        m_binormal = cross(m_normal, m_tangent);
    }

    INLINE HOSTDEVICE void inverseTransform(float3& p) const {
        p = p.x*m_binormal + p.y*m_tangent + p.z*m_normal;
    }

    float3 m_binormal;
    float3 m_tangent;
    float3 m_normal;
};

}