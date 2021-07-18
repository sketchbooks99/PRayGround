#include "scene.h"

namespace oprt {

void Scene::createOnDevice()
{
    
}

// --------------------------------------------------------------------------------
void Scene::cleanUp()
{
    for (auto& ps : m_primitive_instances)
    {
        for (auto& p : ps.primitives())
        {
            std::visit([](auto surface) {
                surface->freeData();
            }, p.surface());
        }
    }
}

// --------------------------------------------------------------------------------
void Scene::createHitgroupPrograms(const Context& ctx, const Module& module)
{
    for (auto& ps : m_primitive_instances)
    {
        for (auto& p : ps.primitives())
        {
            p.createPrograms( ctx, module );
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
                    hitgroup_records.back().data.shape_data = p.shape()->devicePtr();
                    hitgroup_records.back().data.surface_data = 
                        std::visit([](auto surface) { return surface->devicePtr(); }, p.surface());

                    if (p.surfaceType() == SurfaceType::Material)
                    {
                        std::shared_ptr<Material> material = std::get<std::shared_ptr<Material>>(p.surface());
                        hitgroup_records.back().data.surface_func_base_id = 
                            static_cast<uint32_t>(material->type()) * RAY_TYPE_COUNT;
                    }
                    else if (p.surfaceType() == SurfaceType::Emitter)
                    {
                        std::shared_ptr<AreaEmitter> area = std::get<std::shared_ptr<AreaEmitter>>(p.surface());
                        hitgroup_records.back().data.surface_func_base_id = 
                            static_cast<uint32_t>(MaterialType::Count) * RAY_TYPE_COUNT + 
                            static_cast<uint32_t>(TextureType::Count) +
                            static_cast<uint32_t>(area->type());
                    }
                    hitgroup_records.back().data.surface_type = p.surfaceType();
                } 
                // Bind sbt to occlusion program groups.
                else if (i == 1) 
                {
                    hitgroup_records.push_back(HitGroupRecord());
                    p.bindOcclusionRecord(&hitgroup_records.back());
                    hitgroup_records.back().data.shape_data = p.shape()->devicePtr();
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