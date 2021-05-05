#include "scene.h"

namespace oprt {

void Scene::create_hitgroup_programs(const OptixDeviceContext& ctx, const Module& module)
{
    for (auto& ps : m_primitive_instances)
    {
        for (auto& p : ps.primitives())
        {
            p.create_programs(ctx, (OptixModule)module);
        }
    }
}

std::vector<ProgramGroup> Scene::hitgroup_programs()
{
    std::vector<ProgramGroup> program_groups;
    for (auto &ps : m_primitive_instances) {
        for (auto &p : ps.primitives()) {
            std::vector<ProgramGroup> programs = p.program_groups();
            std::copy(programs.begin(), programs.end(), std::back_inserter(program_groups));
        }
    }
    return program_groups;
}

void Scene::create_hitgroup_sbt(OptixShaderBindingTable& sbt) {
    size_t hitgroup_record_size = sizeof(HitGroupRecord);
    std::vector<HitGroupRecord> hitgroup_records;
    for (auto &ps : m_primitive_instances) {
        for (auto &p : ps.primitives()) {
            for (int i=0; i<RAY_TYPE_COUNT; i++) {
                // Bind sbt to radiance program groups. 
                if (i == 0) 
                {
                    hitgroup_records.push_back(HitGroupRecord());
                    p.bind_radiance_record(&hitgroup_records.back());
                    hitgroup_records.back().data.shapedata = p.shape()->get_dptr();
                    hitgroup_records.back().data.matdata = p.material()->get_dptr();
                    hitgroup_records.back().data.material_type = (unsigned int)p.materialtype();
                } 
                // Bind sbt to occlusion program groups.
                else if (i == 1) 
                {
                    hitgroup_records.push_back(HitGroupRecord());
                    p.bind_occlusion_record(&hitgroup_records.back());
                    hitgroup_records.back().data.shapedata = p.shape()->get_dptr();
                }
            }
        }
    }

    CUDABuffer<HitGroupRecord> d_hitgroup_records;
    d_hitgroup_records.alloc_copy(hitgroup_records);

    sbt.hitgroupRecordBase = d_hitgroup_records.d_ptr();
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_records.size());
}

}