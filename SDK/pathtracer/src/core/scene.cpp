#pragma once 

#include <core/scene.h>
#include <optix/sbt.h>

namespace pt {

void Scene::create_hitgroup_programs(const OptixDeviceContext& ctx, const OptixModule& module) {
    for (auto& p : m_primitives)
        p.create_programs(ctx, module);
}

void Scene::create_hitgroup_sbt(const OptixModule& module, OptixShaderBindingTable& sbt) {
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    std::vector<HitGroupRecord> hitgroup_records(RAY_TYPE_COUNT * m_primitives.size());
    int sbt_idx = 0;
    for (auto& p : m_primitives) {
        // Bind HitGroupData to radiance program. 
        hitgroup_records[sbt_idx].data.shapedata = p.shape()->get_dptr();
        hitgroup_records[sbt_idx].data.matptr = (MaterialPtr)p.material()->get_dptr();
        p.bind_radiance_record(hitgroup_records[sbt_idx]);

        // Bind HitGroupData to occlusion program. 
        sbt_idx++;
        memset(&hitgroup_records[sbt_idx], 0, hitgroup_record_size);
        p.bind_occlusion_record(hitgroup_records[sbt_idx]);
    }
}

}