#pragma once

#include <optix.h>
#include <prayground/math/vec_math.h>

#ifndef __CUDACC__
#include <prayground/core/stream_helpers.h>
#endif

namespace prayground {

class AABB {
public:
    AABB() : m_min(make_float3(0.f)), m_max(make_float3(0.f)) {}
    AABB(float3 min, float3 max) : m_min(min), m_max(max) {}
    float3 min() const { return m_min; }
    float3 max() const { return m_max; }

    explicit operator OptixAabb() { return {m_min.x, m_min.y, m_min.z, m_max.x, m_max.y, m_max.z}; }

    float surfaceArea() {
        float dx = m_max.x - m_min.x;
        float dy = m_max.y - m_max.y;
        float dz = m_max.z - m_max.z;
        return 2*(dx*dy + dy*dz + dz*dx);
    }

    static AABB merge(AABB box0, AABB box1)
    {
        float3 min_box = make_float3(
            fmin(box0.min().x, box1.min().x),
            fmin(box0.min().y, box1.min().y),
            fmin(box0.min().z, box1.min().z)
        );

        float3 max_box = make_float3(
            fmax(box0.max().x, box1.max().x),
            fmax(box0.max().y, box1.max().y),
            fmax(box0.max().z, box1.max().z)
        );

        return AABB(min_box, max_box);
    }
private:
    float3 m_min, m_max;
};

#ifndef __CUDACC__

inline std::ostream& operator<<(std::ostream& out, const AABB& aabb)
{
    out << "min: " << aabb.min() << ", max: " << aabb.max();
    return out; 
}

#endif

}