#include "scene.h"

namespace oprt {

// --------------------------------------------------------------------------------
void Scene::freeFromDevice()
{
    for (auto& ps : m_primitive_instances)
    {
        for (auto& p : ps.primitives())
        {
            p.material()->freeData();
        }
    }
}

// --------------------------------------------------------------------------------
void Scene::createHitgroupPrograms(const OptixDeviceContext& ctx, const Module& module)
{
    for (auto& ps : m_primitive_instances)
    {
        for (auto& p : ps.primitives())
        {
            p.createPrograms(ctx, (OptixModule)module);
        }
    }
}

// --------------------------------------------------------------------------------
std::vector<ProgramGroup> Scene::hitgroupPrograms()
{
    std::vector<ProgramGroup> program_groups;
    for (auto &ps : m_primitive_instances) {
        for (auto &p : ps.primitives()) {
            std::vector<ProgramGroup> programs = p.programGroups();
            std::copy(programs.begin(), programs.end(), std::back_inserter(program_groups));
        }
    }
    return program_groups;
}

// --------------------------------------------------------------------------------
void Scene::createHitgroupSBT(OptixShaderBindingTable& sbt) {
    size_t hitgroup_record_size = sizeof(HitGroupRecord);
    std::vector<HitGroupRecord> hitgroup_records;
    for (auto &ps : m_primitive_instances) {
        for (auto &p : ps.primitives()) {
            for (int i=0; i<RAY_TYPE_COUNT; i++) {
                // Bind sbt to radiance program groups. 
                if (i == 0) 
                {
                    hitgroup_records.push_back(HitGroupRecord());
                    p.bindRadianceRecord(&hitgroup_records.back());
                    hitgroup_records.back().data.shapedata = p.shape()->devicePtr();
                    hitgroup_records.back().data.matdata = p.material()->devicePtr();
                    hitgroup_records.back().data.material_type = static_cast<unsigned int>(p.materialType());
                } 
                // Bind sbt to occlusion program groups.
                else if (i == 1) 
                {
                    hitgroup_records.push_back(HitGroupRecord());
                    p.bindOcclusionRecord(&hitgroup_records.back());
                    hitgroup_records.back().data.shapedata = p.shape()->devicePtr();
                }
            }
        }
    }

    CUDABuffer<HitGroupRecord> d_hitgroup_records;
    d_hitgroup_records.copyToDevice(hitgroup_records);

    sbt.hitgroupRecordBase = d_hitgroup_records.devicePtr();
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
    sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_records.size());
}

}