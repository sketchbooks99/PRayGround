#pragma once 

#include <unordered_map>

#include <prayground/core/util.h>
#include <prayground/core/shape.h>
#include <prayground/core/material.h>
#include <prayground/core/texture.h>
#include <prayground/core/camera.h>
#include <prayground/core/bitmap.h>

#include <prayground/emitter/area.h>
#include <prayground/emitter/envmap.h>

#include <prayground/optix/pipeline.h>
#include <prayground/optix/sbt.h>
#include <prayground/optix/geometry_accel.h>
#include <prayground/optix/instance_accel.h>
#include <prayground/optix/instance.h>
#include <prayground/optix/transform.h>

namespace prayground {

    template <class _CamT, uint32_t N>
    requires(std::derived_from<_CamT, Camera>)
    class Scene {
    private:
        template <class T>
        using Item = std::pair<std::string, T>;

    public:
        static constexpr uint32_t NRay = N;
        using CamT = _CamT;
        using SBT = pgDefaultSBT<TCam, NRay>;

        struct Settings {
            bool allow_motion;

            // Acceleration settings
            bool allow_accel_compaction;
            bool allow_accel_multilevel;
            bool allow_accel_update;
        };

        Scene();

        void init();
        void init(const Settings& settings);

        template <class LaunchParams>
        void launchRay(const Context& ctx, const Pipeline& ppl, const LaunchParams& l_params, 
            uint32_t w, uint32_t h, uint32_t d);

        void setupRaygen(ProgramGroup& rg_prg);

        void addBitmap(const std::string& name, PixelFormat fmt, int32_t w, int32_t h);
        void addFloatBitmap(const std::string& name, PixelFormat fmt, int32_t w, int32_t h);

        void setCamera(const _CamT& camera);
        void camera() const;

        void addObject(const std::string& name, Item<std::shared_ptr<Shape>> shape, Item<std::shared_ptr<Material>> material, 
            const Matrix4f& transform = Matrix4f::identity());
        void addLightObject(const std::string& name, Item<std::shared_ptr<Shape>> shape, Item<std::shared_ptr<AreaEmitter>> area,
            const Matrix4f& transform = Matrix4f::identity());

        void addMovingObject(const std::string& name, Item<std::shared_ptr<Shape>> shape, Item<std::shared_ptr<Material> material,
            const Matrix4f& begin_transform = Matrix4f::identity(), const Matrix4f& end_transform = Matrix4f::identity(), uint16_t num_key = 2);
        void addMovingLightObject(const std::string& name, Item<std::shared_ptr<Shape>> shape, Item<std::shared_ptr<AreaEmitter> area,
            const Matrix4f& begin_transform = Matrix4f::identity(), const Matrix4f& end_transform = Matrix4f::identity(), uint16_t num_key = 2);

        void createOnDevice();

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
            uint16_t num_key;
            Obj object;
            Matrix4f end_transform;
        };

        // Optix states
        Context                     m_ctx;
        Pipeline                    m_pipeline;
        SBT                         m_sbt;
        std::vector<InstanceAccel>  m_accel;    // m_accel[0] -> Top level

        // Rendered results
        std::unordered_map<Item<Bitmap>>        m_bitmaps;
        std::unordered_map<Item<FloatBitmap>>   m_fbitmaps;

        // Camera
        CamT m_camera;

        // Environement emitter
        std::shared_ptr<EnvironmentEmitter> m_envmap;

        // Objects
        std::unordered_map<std::string, Object>         m_objects; 

        // Area lights
        std::unordered_map<std::string, LightObject>    m_light_objects;
    };

    template <class _CamT, uint32_t N>
    inline Scene<_CamT, N>::Scene()
    {

    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::init()
    {

    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::init(const Scene::Settings& settings)
    {

    }

    template <class _CamT, uint32_t N>
    template <class LaunchParams>
    inline void Scene<_CamT, N>::launchRay(
        const Context& ctx, const Pipeline& ppl, const LaunchParams& l_params, 
        uint32_t w, uint32_t h, uint32_t d)
    {
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addBitmap(const std::string& name, PixelFormat fmt, int32_t w, int32_t h)
    {
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addFloatBitmap(const std::string& name, PixelFormat fmt, int32_t w, int32_t h)
    {
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::setCamera(const _CamT& camera)
    {
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::camera() const
    {
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addObject(const std::string& name, Item<std::shared_ptr<Shape>> shape, Item<std::shared_ptr<Material>> material, const Matrix4f& transform)
    {
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addLightObject(const std::string& name, Item<std::shared_ptr<Shape>> shape, Item<std::shared_ptr<AreaEmitter>> area, const Matrix4f& transform)
    {
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingObject(const std::string& name, Item<std::shared_ptr<Shape>> shape, Item<std::shared_ptr<Material> material, const Matrix4f& begin_transform = Matrix4f::identity(), const Matrix4f& end_transform = Matrix4f::identity(), uint16_t num_key = 2)
    {
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingLightObject(const std::string& name, Item<std::shared_ptr<Shape>> shape, Item<std::shared_ptr<AreaEmitter> area, const Matrix4f& begin_transform = Matrix4f::identity(), const Matrix4f& end_transform = Matrix4f::identity(), uint16_t num_key = 2)
    {
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::createOnDevice()
    {
    }

} // namespace prayground