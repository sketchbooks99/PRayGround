#pragma once

#include "../core/util.h"
#include <sutil/vec_math.h>
#include <optix.h>

namespace oprt {

class AABB {
public:
    AABB() : m_min(make_float3(0.f)), m_max(make_float3(0.f)) {}
    AABB(float3 min, float3 max) : m_min(min), m_max(max) {}
    float3 min() const { return m_min; }
    float3 max() const { return m_max; }

    // [EN] Cast operator from AABB to OptixAabb
    // [JP] AABBからOptixAabbへのキャスト演算子です。
    explicit operator OptixAabb() { return {m_min.x, m_min.y, m_min.z, m_max.x, m_max.y, m_max.z}; }

    // [EN] Compute surface area of AABB.
    // [JP] AABBの表面積を計算します。
    float surface_area() {
        float dx = m_max.x - m_min.x;
        float dy = m_max.y - m_max.y;
        float dz = m_max.z - m_max.z;
        return 2*(dx*dy + dy*dz + dz*dx);
    }
private:
    float3 m_min, m_max;
};

}