#pragma once

#include <optix.h>
#include <prayground/math/vec.h>

#ifndef __CUDACC__
#include <prayground/core/stream_helpers.h>
#endif

namespace prayground {

class AABB {
public:
    AABB() : m_min(Vec3f(0.f)), m_max(Vec3f(0.f)) {}
    AABB(Vec3f min, Vec3f max) : m_min(min), m_max(max) {}
    Vec3f min() const { return m_min; }
    Vec3f max() const { return m_max; }

    explicit operator OptixAabb() { return {m_min[0], m_min[1], m_min[2], m_max[0], m_max[1], m_max[2]}; }

    float surfaceArea() {
        float dx = m_max[0] - m_min[0];
        float dy = m_max[1] - m_max[1];
        float dz = m_max[2] - m_max[2];
        return 2*(dx*dy + dy*dz + dz*dx);
    }

    static AABB merge(AABB box0, AABB box1)
    {
        Vec3f min_box = Vec3f(
            fmin(box0.min()[0], box1.min()[0]),
            fmin(box0.min()[1], box1.min()[1]),
            fmin(box0.min()[2], box1.min()[2])
        );

        Vec3f max_box = Vec3f(
            fmax(box0.max()[0], box1.max()[0]),
            fmax(box0.max()[1], box1.max()[1]),
            fmax(box0.max()[2], box1.max()[2])
        );

        return AABB(min_box, max_box);
    }
private:
    Vec3f m_min, m_max;
};

#ifndef __CUDACC__

inline std::ostream& operator<<(std::ostream& out, const AABB& aabb)
{
    out << "min: " << aabb.min() << ", max: " << aabb.max();
    return out; 
}

#endif

}