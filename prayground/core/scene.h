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
        // ID is for shader binding table offset
        template <class T>
        struct Item {
            std::string name;
            uint32_t ID;
            T value;
        };

        template <class T>
        using Pair = std::pair<std::string, T>;

    public:
        static constexpr uint32_t NRay = N;
        using CamT = _CamT;
        using SBT = pgDefaultSBT<CamT, NRay>;

        template <class SurfaceT>
        struct Object_ {
            Pair<std::shared_ptr<Shape>> shape;
            Pair<std::shared_ptr<SurfaceT>> surface;

            SurfaceCallableID surface_callable_id;

            Matrix4f transform;
        };

        template <class SurfaceT>
        struct MovingObject_ {
            Pair<std::shared_ptr<Shape>> shape;
            Pair<std::shared_ptr<SurfaceT>> surface;

            SurfaceCallableID surface_callable_id;

            Matrix4f begin_transform;
            Matrix4f end_transform;
            uint32_t num_key = 2;
        };

        using Object = Object_<Material>;
        using LightObject = Object_<AreaEmitter>;
        using MovingObject = MovingObject_<Material>;
        using MovingLightObject = MovingObject_<AreaEmitter>;

        struct Settings {
            bool allow_motion;

            // Acceleration settings
            bool allow_accel_compaction;
            bool allow_accel_update;
        };

        Scene();

        void setup();
        void setup(const Settings& settings);

        void update();

        template <class LaunchParams>
        void launchRay(const Context& ctx, const Pipeline& ppl, LaunchParams& l_params, CUstream stream,
            uint32_t w, uint32_t h, uint32_t d);

        void bindRaygen(ProgramGroup& raygen_prg);

        template <ProgramGroup... Prgs>
        void bindMiss(Prgs&... prgs);

        template <ProgramGroup... Prgs>
        void bindProgramWithObject(const std::string& obj_name, Prgs&... prgs);

        void bindCallabes(ProgramGroup& prg);

        void bindException(ProgramGroup& prg);

        void setCamera(const _CamT& camera);
        _CamT& camera();
        const _CamT& camera() const;

        // Automatically load and create envmap texture from file
        void setEnvmap(const std::shared_ptr<Texture>& texture);

        /// @note Should create/deletion functions for object return boolean value?
        // Object
        void addObject(const std::string& name, const Object& object);
        void addObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, Pair<std::shared_ptr<Material>> material, 
            const Matrix4f& transform = Matrix4f::identity());
        void addObject(const std::string& name, const std::string& shape_name, Pair<std::shared_ptr<Material>> material, 
            const Matrix4f& transform = Matrix4f::identity());
        void addObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, const std::string& mat_name, 
            const Matrix4f& transform = Matrix4f::identity());
        void duplicateObject(const std::string& name, const Matrix4f& transform = Matrix4f::identity());

        // Light object
        void addLightObject(const std::string& name, const LightObject& light_object);
        void addLightObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, Pair<std::shared_ptr<AreaEmitter>> light,
            const Matrix4f& transform = Matrix4f::identity());
        void addLightObject(const std::string& name, const std::string& shape_name, Pair<std::shared_ptr<AreaEmitter>> light, 
            const Matrix4f& transform = Matrix4f::identity());
        void addLightObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, const std::string& light_name, 
            const Matrix4f& transform = Matrix4f::identity());
        void duplicateLightObject(const std::string& name, const Matrix4f& transform = Matrix4f::identity());

        void addMovingObject(const std::string& name, const MovingObject& moving_object);
        void addMovingObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, Pair<std::shared_ptr<Material>> material,
            const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void addMovingObject(const std::string& name, const std::string& shape_name, Pair<std::shared_ptr<Material>> material, 
            const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void addMovingObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, const std::string& mat_name, 
            const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void duplicateMovingObject(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);

        void addMovingLightObject(const std::string& name, const MovingLightObject& moving_light_object);
        void addMovingLightObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, Pair<std::shared_ptr<AreaEmitter>> light,
            const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void addMovingLightObject(const std::string& name, const std::string& shape_name, Pair<std::shared_ptr<AreaEmitter>> light,
            const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void addMovingLightObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, const std::string& light_name,
            const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void duplicateMovingLightObject(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);

        // Erase object and corresponding shader binding table record
        void deleteObject(const std::string& name);

        void buildAccel(const Context& ctx, CUstream stream);
        void buildSBT(const Context& ctx);
    private:
        template <class T>
        static std::optional<Item<T>> findItem(const std::vector<Item<T>>& items, const std::string& name)
        {
            for (const auto& item : items)
            {
                if (item.name == name)
                    return item;
            }
            return std::nullopt_t;
        }

        template <class Obj>
        static std::optional<Pair<Shape>> findShape(const std::vector<Item<Obj>>& objects, const std::string& shape_name)
        {
            for (const auto& obj : objects)
            {
                if (obj.shape.first == shape_name)
                    return obj.shape;
            }
            return std::nullopt_t;
        }

        template <class Obj, class ReturnT>
        static std::optional<Pair<ReturnT>> findSurface(const std::vector<Item<Obj>>& objects, const std::string& mat_name)
        {
            for (const auto& obj : objects)
            {
                if (obj.surface.first == mat_name)
                    return obj.surface;
            }
            return std::nullopt_t;
        }

        template <typename Obj>
        struct MovingObject {
            uint16_t num_key;
            Obj object;
            Matrix4f end_transform;
        };

        struct SBTUpdateState {
            bool raygen_updated;
            bool miss_updated;
            bool hitgroup_updated;
        };

        Settings m_settings;

        SBT                         m_sbt;      // Shader binding table
        SBTUpdateState              m_sbt_update_state;
        uint32_t                    m_sbt_end_id;
        std::vector<InstanceAccel>  m_accel;    // m_accel[0] -> Top level
        CUDABuffer<void>            d_params;   // Data region on device side for OptixLaunchParams

        // Camera
        CamT m_camera;

        // Environement emitter
        std::shared_ptr<EnvironmentEmitter> m_envmap;

        // Objects
        std::vector<Item<Object>>                     m_objects; 
        std::vector<Item<MovingObject<Object>>>       m_moving_objects;

        // Area lights
        std::vector<Item<LightObject>>                m_light_objects;
        std::vector<Item<MovingObject<LightObject>>>  m_moving_light_objects;
    };

    template <class _CamT, uint32_t N>
    inline Scene<_CamT, N>::Scene()
    {
        
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::setup()
    {
        Settings settings;
        settings.allow_motion           = false;
        settings.allow_accel_compaction = true;
        settings.allow_accel_update     = false;
        this->init(settings);
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::setup(const Scene::Settings& settings)
    {
        m_settings = settings;
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::update()
    {
        UNIMPLEMENTED();
    }

    template <class _CamT, uint32_t N>
    template <class LaunchParams>
    inline void Scene<_CamT, N>::launchRay(
        const Context& ctx, const Pipeline& ppl, LaunchParams& l_params, CUstream steram,
        uint32_t w, uint32_t h, uint32_t d)
    {
        if (!d_params.isAllocated() || d_params.size() < sizeof(LaunchParams))
            d_params.allocate(sizeof(LaunchParams));
        optixLaunch(
            static_cast<OptixPipeline>(ppl), stream,
            d_params.devicePtr(), d_params.size(), m_sbt.sbt(),
            w, h, d);
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::bindRaygen(ProgramGroup& rg_prg)
    {
        RaygenRecord& rg_record = m_sbt.raygenRecord();
        rg_prg.recordPackHeader(&rg_record);
    }

    template<class _CamT, uint32_t N>
    template<ProgramGroup ...Prgs>
    inline void Scene<_CamT, N>::bindMiss(Prgs & ...prgs)
    {

    }

    template<class _CamT, uint32_t N>
    template<ProgramGroup ...Prgs>
    inline void Scene<_CamT, N>::bindProgramWithObject(const std::string& obj_name, Prgs & ...prgs)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::bindCallabes(ProgramGroup& prg)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::bindException(ProgramGroup& prg)
    {
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::setCamera(const _CamT& camera)
    {
        m_camera = camera;
    }

    template<class _CamT, uint32_t N>
    inline _CamT& Scene<_CamT, N>::camera()
    {
        return camera;
    }

    template <class _CamT, uint32_t N>
    inline const _CamT& Scene<_CamT, N>::camera() const
    {
        return camera;
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::setEnvmap(const std::shared_ptr<Texture>& texture)
    {
        if (!m_envmap) 
            m_envmap = make_shared<EnvironmentEmitter>(texture);
        else
        {
            m_sbt_update_state.miss_updated = true;
            m_envmap->setTexture(texture);
        }
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addObject(const std::string& name, const Object& object)
    {
        m_objects.emplace_back({name, })
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, Pair<std::shared_ptr<Material>> material, const Matrix4f& transform)
    {
        m_objects.emplace_back({ name, shape, material, transform });
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addObject(const std::string& name, const std::string& shape_name, Pair<std::shared_ptr<Material>> material, const Matrix4f& transform)
    {
        auto addObjectWithExistingShape = [&](const auto& objects) -> bool
        {
            auto shape = findShape(objects);
            if (!shape)
                return false;
            addObject(name, shape, material, transform);
            return true;
        };

        if (addObjectWithExistingShape(m_objects)) return;
        if (addObjectWithExistingShape(m_light_objects)) return;
        if (addObjectWithExistingShape(m_moving_objects)) return;
        if (addObjectWithExistingShape(m_moving_light_objects)) return;

        pgLogFatal()
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, const std::string& mat_name, const Matrix4f& transform)
    {
        {
            auto material = findSurface<Item<Object>, std::shared_ptr<Material>>(m_objects, mat_name);
            if (material)
            {
                addObject(name, shape, material, transform);
                return;
            }
        }

        {
            auto material = findSurface<Item<MovingObject>, std::shared_ptr<Material>>(m_moving_objects, mat_name);
            if (material)
            {
                addObject(name, shape, material, transform);
                return;
            }
        }
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::duplicateObject(const std::string& name, const Matrix4f& transform)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addLightObject(const std::string& name, const LightObject& light_object)
    {
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addLightObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, Pair<std::shared_ptr<AreaEmitter>> area, const Matrix4f& transform)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addLightObject(const std::string& name, const std::string& shape_name, Pair<std::shared_ptr<AreaEmitter>> light, const Matrix4f& transform)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addLightObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, const std::string& light_name, const Matrix4f& transform)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::duplicateLightObject(const std::string& name, const Matrix4f& transform)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingObject(const std::string& name, const MovingObject& moving_object)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, Pair<std::shared_ptr<Material>> material, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingObject(const std::string& name, const std::string& shape_name, Pair<std::shared_ptr<Material>> material, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, const std::string& mat_name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::duplicateMovingObject(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingLightObject(const std::string& name, const MovingLightObject& moving_light_object)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingLightObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, Pair<std::shared_ptr<AreaEmitter>> light, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingLightObject(const std::string& name, const std::string& shape_name, Pair<std::shared_ptr<AreaEmitter>> light, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingLightObject(const std::string& name, Pair<std::shared_ptr<Shape>> shape, const std::string& light_name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::duplicateMovingLightObject(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::deleteObject(const std::string& name)
    {
    }

    template<class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::buildAccel(const Context& ctx, CUstream stream)
    {
    }

    template <class _CamT, uint32_t N>
    inline void Scene<_CamT, N>::buildSBT(const Context& ctx)
    {
    }

} // namespace prayground