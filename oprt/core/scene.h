#pragma once

#include <oprt/core/bitmap.h>
#include <oprt/core/shape.h>
#include <oprt/core/material.h>
#include <oprt/optix/context.h>
#include <oprt/optix/pipeline.h>
#include <oprt/optix/shape_instance.h>
#include <oprt/optix/accel.h>
#include <oprt/emitter/area.h>
#include <oprt/emitter/envmap.h>
#include <unordered_map>
// #include "../oprt.h"

namespace oprt {

//class Scene {
//public:
//    Scene() {}
//
//    void createOnDevice();
//    void cleanUp();
//
//    void render();
//
//    /** 
//     * @brief Create programs associated with primitives.
//     */
//    void createHitgroupPrograms(const Context& ctx, const Module& module);
//
//    /**
//     * @brief Return all hitgroup programs contained in Scene
//     */
//    std::vector<ProgramGroup> hitgroupPrograms();
//
//    /** 
//     * @brief Create SBT with HitGroupData. 
//     * @note SBTs for raygen and miss program aren't created at here.
//     */
//    void createHitgroupSBT(OptixShaderBindingTable& sbt);
//
//    void addPrimitiveInstance(PrimitiveInstance ps) {
//        ps.sort();
//        if (m_primitive_instances.empty())
//            ps.setSbtIndexBase(0);
//        else
//            ps.setSbtIndexBase(m_primitive_instances.back().sbtIndex());
//        m_primitive_instances.push_back(ps); 
//    }
//    std::vector<PrimitiveInstance> primitiveInstances() const { return m_primitive_instances; }
//
//    void setEnvironment(const float3& color) 
//    { 
//        m_environment = std::make_shared<EnvironmentEmitter>(color);
//    }
//    void setEnvironment(const std::shared_ptr<Texture>& texture)
//    {
//        m_environment = std::make_shared<EnvironmentEmitter>(texture);
//    }
//    void setEnvironment(const std::shared_ptr<EnvironmentEmitter>& env) { m_environment = env; }
//    void setEnvironment(const std::filesystem::path& filename)
//    {
//        m_environment = std::make_shared<EnvironmentEmitter>(filename);
//    }
//    std::shared_ptr<EnvironmentEmitter> environment() const { return m_environment; }
//private:
//    std::vector<PrimitiveInstance> m_primitive_instances;   // Primitive instances with same transformation.
//    std::shared_ptr<EnvironmentEmitter> m_environment;      // Environment map
//};

/**
 * RTScene �� Shader Binding Table �Ƃ͓Ɨ��ɊǗ�����B
 * �����܂ŁA���C�g���[�V���O����ΏۂƂȂ�V�[��
 * (Environment Emitter, Shape, Material, Texture) ���Ǘ�����݂̂ɂƂǂ܂�
 */
template <class Key>
class RTScene {
public:
    RTScene(const uint32_t num_ray_type);

    void createDataOnDevice();
    void buildAccelStructure();
    void updateAccelStructure();
    void destroy();

    void addShape(const Key& instance_key, const Key& shape_key, const std::shared_ptr<Shape>& shape);
    void addInstance(const Key& instance_key, const std::shared_ptr<Instance>& instance);
    void addShapeInstance(const Key& key, const std::shared_ptr<ShapeInstance>& shape_instance);
    std::shared_ptr<ShapeInstance> getShapeInstance(const Key& key) const;
    std::shared_ptr<Shape> getShape(const Key& instance_key, const Key& shape_key) const;

    // Future work
    // void eraseShapeInstance(const Key& name);
    // void eraseShapeFromInstance(const Key& instance_name, const Key& shape_name) const;

    void addMaterial(const Key& key, const std::shared_ptr<Material>& material);
    std::shared_ptr<Material> getMaterial(const Key& key) const;

    void addTexture(const Key& key, const std::shared_ptr<Texture>& texture);
    std::shared_ptr<Texture> getTexture(const Key& key) const;

    OptixTraversableHandle handle() const;
private:
    std::shared_ptr<EnvironmentEmitter> m_enviroment;
    std::unordered_map<Key, std::shared_ptr<Instance>> m_instances;
    std::unordered_map<Key, std::shared_ptr<Material>> m_materials;
    std::unordered_map<Key, std::shared_ptr<Texture>> m_textures;
    InstanceAccel m_instance_accel;
    uint32_t m_num_ray_type{ 1 };
};

}