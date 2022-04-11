#pragma once 

#include <unordered_map>

#include <prayground/core/util.h>
#include <prayground/core/shape.h>
#include <prayground/core/material.h>
#include <prayground/core/texture.h>
#include <prayground/core/camera.h>

#include <prayground/emitter/area.h>
#include <prayground/emitter/envmap.h>

#include <prayground/optix/pipeline.h>
#include <prayground/optix/sbt.h>
#include <prayground/optix/geometry_accel.h>
#include <prayground/optix/instance_accel.h>
#include <prayground/optix/instance.h>
#include <prayground/optix/transform.h>

namespace prayground {

    template <class _TCam, class _LaunchParams, uint32_t N>
    requires(std::derived_from<TCam, Camera>)
    class Scene {
    private:
        template <class T>
        using Item = std::pair<std::string, T>;

    public:
        static constexpr uint32_t NRay = N;
        using TCam = _TCam;
        using LaunchParams = _LaunchParams;
        using SBT = pgDefaultSBT<TCam, NRay>;

        Scene();

        void setCamera(const _TCam& camera);

        void addObject(Item<std::shared_ptr<Shape>> shape, Item<std::shared_ptr<Material>> material);
        void addLightObject(Item<std::shared_ptr<Shape>> shape, Item<std::shared_ptr<Material>> material);

    private:
        struct Object {
            std::string name;
            Item<std::shared_ptr<Shape>> shape;
            Item<std::shared_ptr<Material>> material;
            
            Matrix4f transform;
        };

        struct LightObject {
            std::string name;
            Item<std::shared_ptr<Shape>> shape;
            Item<std::shared_ptr<AreaEmitter>> light;

            Matrix4f transform;
        };

        template <typename Obj>
        struct MovingObject {
            Obj object;
            Matrix4f end_transform;
        };

        // Optix states
        Context  m_ctx;
        Pipeline m_pipeline;
        SBT      m_sbt;

        // Camera
        TCam m_camera;

        // Environement emitter
        std::shared_ptr<EnvironmentEmitter> env;

        // Objects
        std::unordered_map<std::string, Object>         m_objects; 

        // Area lights
        std::unordered_map<std::string, LightObject>    m_light_objects;
    };

} // namespace prayground