#include "scene.h"

namespace oprt {

//void Scene::createOnDevice()
//{
//    
//}
//
//// --------------------------------------------------------------------------------
//void Scene::cleanUp()
//{
//    for (auto& ps : m_primitive_instances)
//    {
//        for (auto& p : ps.primitives())
//        {
//            std::visit([](auto surface) {
//                surface->freeData();
//            }, p.surface());
//        }
//    }
//}
//
//// --------------------------------------------------------------------------------
//void Scene::createHitgroupPrograms(const Context& ctx, const Module& module)
//{
//    for (auto& ps : m_primitive_instances)
//    {
//        for (auto& p : ps.primitives())
//        {
//            p.createPrograms( ctx, module );
//        }
//    }
//}
//
//// --------------------------------------------------------------------------------
//std::vector<ProgramGroup> Scene::hitgroupPrograms()
//{
//    std::vector<ProgramGroup> program_groups;
//    for (auto &ps : m_primitive_instances) {
//        for (auto &p : ps.primitives()) {
//            std::vector<ProgramGroup> programs = p.programGroups();
//            std::copy(programs.begin(), programs.end(), std::back_inserter(program_groups));
//        }
//    }
//    return program_groups;
//}
//
// --------------------------------------------------------------------------------
//void Scene::createHitgroupSBT(OptixShaderBindingTable& sbt) {
//    size_t hitgroup_record_size = sizeof(HitGroupRecord);
//    std::vector<HitGroupRecord> hitgroup_records;
//    for (auto &ps : m_primitive_instances) {
//        for (auto &p : ps.primitives()) {
//            for (int i=0; i<RAY_TYPE_COUNT; i++) {
//                // Bind sbt to radiance program groups. 
//                if (i == 0) 
//                {
//                    hitgroup_records.push_back(HitGroupRecord());
//                    p.bindRadianceRecord(&hitgroup_records.back());
//                    hitgroup_records.back().data.shape_data = p.shape()->devicePtr();
//                    hitgroup_records.back().data.surface_data = 
//                        std::visit([](auto surface) { return surface->devicePtr(); }, p.surface());
//
//                    if (p.surfaceType() == SurfaceType::Material)
//                    {
//                        std::shared_ptr<Material> material = std::get<std::shared_ptr<Material>>(p.surface());
//                        hitgroup_records.back().data.surface_func_base_id = 
//                            static_cast<uint32_t>(material->type()) * RAY_TYPE_COUNT;
//                    }
//                    else if (p.surfaceType() == SurfaceType::Emitter)
//                    {
//                        std::shared_ptr<AreaEmitter> area = std::get<std::shared_ptr<AreaEmitter>>(p.surface());
//                        hitgroup_records.back().data.surface_func_base_id = 
//                            static_cast<uint32_t>(MaterialType::Count) * RAY_TYPE_COUNT + 
//                            static_cast<uint32_t>(TextureType::Count) +
//                            static_cast<uint32_t>(area->type());
//                    }
//                    hitgroup_records.back().data.surface_type = p.surfaceType();
//                } 
//                // Bind sbt to occlusion program groups.
//                else if (i == 1) 
//                {
//                    hitgroup_records.push_back(HitGroupRecord());
//                    p.bindOcclusionRecord(&hitgroup_records.back());
//                    hitgroup_records.back().data.shape_data = p.shape()->devicePtr();
//                }
//            }
//        }
//    }
//
//    CUDABuffer<HitGroupRecord> d_hitgroup_records;
//    d_hitgroup_records.copyToDevice(hitgroup_records);
//
//    sbt.hitgroupRecordBase = d_hitgroup_records.devicePtr();
//    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
//    sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_records.size());
//}

// --------------------------------------------------------------------------------
RTScene::RTScene(const uint32_t num_ray_type)
: m_num_ray_type(num_ray_type)
{

}

// --------------------------------------------------------------------------------
void RTScene::createDataOnDevice()
{
	for (auto& shape_instance : m_shape_instances)
		shape_instance.second->copyToDevice();

	for (auto& material : m_materials) {
		if (!material.second->devicePtr()) 
			material.second->copyToDevice();
	}

	for (auto& texture : m_textures) {
		if (!texture.second->devicePtr())
			texture.second->copyToDevice();
	}
}

void RTScene::buildAccelStructure()
{
	// Create geometry acceleration structures for shape
	for (auto& shape_instance : m_shape_instances)
		shape_instance.second->buildAccelStructure();
	
}

void RTScene::updateAccelStructure()
{

}

// --------------------------------------------------------------------------------
void RTScene::destroy()
{

}

// --------------------------------------------------------------------------------
void RTScene::addShape(const std::string& instance_name, const std::string& shape_name, const std::shared_ptr<Shape>& shape)
{

}

void RTScene::addShapeInstance(const std::string& name, const std::shared_ptr<Shape>& shape)
{

}

//void RTScene::eraseShapeInstance(const std::string& name)
//{
//
//}
//
//void RTScene::eraseShapeFromInstance(const std::string& instance_name, const std::string& shape_name) const
//{
//
//}

std::shared_ptr<ShapeInstance> RTScene::getShapeInstance(const std::string& name) const
{

}

std::shared_ptr<Shape> RTScene::getShape(const std::string& instance_name, const std::string& shape_name) const
{

}

// --------------------------------------------------------------------------------
void RTScene::addMaterial(const std::string& name, const std::shared_ptr<Material>& material)
{

}

std::shared_ptr<Material> RTScene::getMaterial(const std::string& name) const
{

}

// --------------------------------------------------------------------------------
void RTScene::addTexture(const std::string& name, const std::shared_ptr<Texture>& texture)
{

}

std::shared_ptr<Texture> RTScene::getTexture(const std::string& name) const
{

}

// --------------------------------------------------------------------------------
OptixTraversableHandle RTScene::handle() const
{

}

} // ::oprt