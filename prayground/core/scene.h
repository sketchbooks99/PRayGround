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

#include <prayground/shape/trianglemesh.h>

namespace prayground {
    template <class T>
    concept DerivedFromCamera = std::derived_from<T, Camera>;

    template <DerivedFromCamera _CamT, uint32_t N>
    class Scene {
    private:
        template <class T>
        struct Item {
            std::string name;
            uint32_t ID; // Used for shader binding table offset
            T value;     
        };

        struct Object {
            std::shared_ptr<Shape> shape;
            std::vector<std::shared_ptr<Material>> materials;
            ShapeInstance instance;
        };

        struct MovingObject {
            std::shared_ptr<Shape> shape;
            std::vector<std::shared_ptr<Material>> materials;

            Instance instance;
            GeometryAccel gas;
            Transform matrix_transform;
        };

        struct Light {
            std::shared_ptr<Shape> shape;
            std::shared_ptr<AreaEmitter> emitter;
            ShapeInstance instance;
        };

        struct MovingLight {
            std::shared_ptr<Shape> shape;
            std::shared_ptr<AreaEmitter> emitter;

            Instance instance;
            GeometryAccel gas;
            Transform matrix_transform;
        };
    public:
        static constexpr uint32_t NRay = N;
        using CamT = _CamT;
        using SBT = pgDefaultSBT<CamT, NRay>;

        struct Settings {
            bool allow_motion;

            // Acceleration settings
            bool allow_accel_compaction;
            bool allow_accel_update;
        };

        Scene();

        void setup();
        void setup(const Settings& settings);

        template <class LaunchParams>
        void launchRay(const Context& ctx, const Pipeline& ppl, LaunchParams& l_params, CUstream stream,
            uint32_t w, uint32_t h, uint32_t d);

        // Bind program to SBT parameters
        void bindRaygenProgram(ProgramGroup& raygen_prg);
        void bindMissPrograms(std::array<ProgramGroup, N>& miss_prgs);
        void bindCallablesProgram(ProgramGroup& prg);
        void bindExceptionProgram(ProgramGroup& prg);

        void setCamera(const std::shared_ptr<_CamT>& camera);
        const std::shared_ptr<_CamT>& camera();

        // Automatically load and create envmap texture from file
        void setEnvmap(const std::shared_ptr<Texture>& texture);

        /// @note Should create/deletion functions for object return boolean value?
        // Object
        void addObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<Material> material, 
            std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& transform = Matrix4f::identity());
        void addObject(const std::string& name, std::shared_ptr<Shape> shape, const std::vector<std::shared_ptr<Material>>& materials,
            std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& transform = Matrix4f::identity());
        void duplicateObject(const std::string& orig_name, const std::string& name, const Matrix4f& transform = Matrix4f::identity());
        void updateObjectTransform(const std::string& name, const Matrix4f& transform);
        bool deleteObject(const std::string& name);

        // Light object
        void addLight(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<AreaEmitter> light,
            std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& transform = Matrix4f::identity());
        void duplicateLight(const std::string& orig_name, const std::string& name, const Matrix4f& transform = Matrix4f::identity());
        void updateLightTransform(const std::string& name, const Matrix4f& transform);
        bool deleteLight(const std::string& name);

        // Moving object (especially for motion blur)
        void addMovingObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<Material> material,
            std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void addMovingObject(const std::string& name, std::shared_ptr<Shape> shape, const std::vector<std::shared_ptr<Material>>& materials,
            std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void duplicateMovingObject(const std::string& orig_name, const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void updateMovingObjectTransform(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform);
        bool deleteMovingObject(const std::string& name);

        // Moving light (especially for motion blur)
        void addMovingLight(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<AreaEmitter> light,
            std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void duplicateMovingLight(const std::string& orig_name, const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void updateMovingLightTransform(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform);
        bool deleteMovingLight(const std::string& name);

        // Collect area emitters from whole lights
        std::vector<std::shared_ptr<AreaEmitter>> areaEmitters() const;

        // The total number of lights contains light and moving_lights
        uint32_t numLights() const;

        void copyDataToDevice();

        void buildAccel(const Context& ctx, CUstream stream);
        void updateAccel(const Context& ctx, CUstream stream);
        OptixTraversableHandle accelHandle() const;

        void buildSBT();
        void updateSBT(uint32_t record_type);
    private:
        template <class T>
        static std::optional<Item<T>> findItem(const std::vector<Item<T>>& items, const std::string& name)
        {
            for (const auto& item : items)
            {
                if (item.name == name)
                    return item;
            }
            return std::nullopt;
        }

        template <class T>
        static std::optional<Item<T>> deleteItem(std::vector<Item<T>>& items, const std::string& name)
        {
            for (auto it = items.begin(); it != items.end();)
            {
                if (it->name == name)
                {
                    items.erase(it);
                    return *it;
                }
                else
                {
                    it++;
                }
            }
        }

        Settings m_settings;

        SBT                         m_sbt;          // Shader binding table
        uint32_t                    m_current_sbt_id;
        InstanceAccel               m_accel;        // m_accel[0] -> Top level
        CUDABuffer<void>            d_params;       // Data region on device side for OptixLaunchParams

        // Camera
        std::shared_ptr<CamT> m_camera;

        // Environement emitter
        std::shared_ptr<EnvironmentEmitter>   m_envmap;

        // Objects
        std::vector<Item<Object>>             m_objects;
        std::vector<Item<MovingObject>>       m_moving_objects;

        // Area lights
        std::vector<Item<Light>>        m_lights;
        std::vector<Item<MovingLight>>  m_moving_lights;

        // Flag represents scene states should be updated.
        bool should_accel_updated;
        bool should_sbt_updated;
    };

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t N>
    inline Scene<_CamT, N>::Scene()
    {
        this->setup();
    }

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::setup()
    {
        Settings settings;
        settings.allow_motion           = false;
        settings.allow_accel_compaction = true;
        settings.allow_accel_update     = false;
        this->setup(settings);
    }

    template <DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::setup(const Scene::Settings& settings)
    {
        m_settings = settings;

        // Initialize instance acceleration structure
        m_accel = InstanceAccel{ InstanceAccel::Type::Instances };

        // Initialize raygen and miss record
        pgRaygenRecord<_CamT> rg_record = {};
        m_sbt.setRaygenRecord(rg_record);
        
        std::array<pgMissRecord, N> ms_records{};
        m_sbt.setMissRecord(ms_records);
    }

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t N>
    template <class LaunchParams>
    inline void Scene<_CamT, N>::launchRay(
        const Context& ctx, const Pipeline& ppl, LaunchParams& l_params, CUstream stream,
        uint32_t w, uint32_t h, uint32_t d)
    {
        // Check if launch parameter is allocated on device
        if (!d_params.isAllocated() || d_params.size() < sizeof(LaunchParams))
            d_params.allocate(sizeof(LaunchParams));

        // Copy a launch parameter to device
        d_params.copyToDeviceAsync(&l_params, sizeof(LaunchParams), stream);

        // Launch raygen kernel on device
        optixLaunch(
            static_cast<OptixPipeline>(ppl), stream,
            d_params.devicePtr(), d_params.size(), &m_sbt.sbt(),
            w, h, d);
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::bindRaygenProgram(ProgramGroup& rg_prg)
    {
        // Fill the record header with the raygen program
        pgRaygenRecord<_CamT>& rg_record = m_sbt.raygenRecord();
        rg_prg.recordPackHeader(&rg_record);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::bindMissPrograms(std::array<ProgramGroup, N>& miss_prgs)
    {
        // Fill record's headers with the miss programs
        for (int i = 0; i < N; i++)
        {
            pgMissRecord& record = m_sbt.missRecord(i);
            miss_prgs[i].recordPackHeader(&record);
        }
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::bindCallablesProgram(ProgramGroup& prg)
    {
        // Add SBT record and fill the record header at the same time
        // since the pgCallablesRecord has no data
        pgCallablesRecord record{};
        prg.recordPackHeader(&record);
        m_sbt.addCallablesRecord(record);
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::bindExceptionProgram(ProgramGroup& prg)
    {
        // Add SBT record and fill the record header at the same time
        // since the pgExceptionRecord has no data
        pgExceptionRecord record{};
        prg.recordPackHeader(&record);
        m_sbt.setExceptionRecord(record);
    }

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::setCamera(const std::shared_ptr<_CamT>& camera)
    {
        m_camera = camera;
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline const std::shared_ptr<_CamT>& Scene<_CamT, N>::camera()
    {
        return m_camera;
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::setEnvmap(const std::shared_ptr<Texture>& texture)
    {
        if (!m_envmap) 
            m_envmap = make_shared<EnvironmentEmitter>(texture);
        else
        {
            m_envmap->setTexture(texture);
        }
    }

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<Material> material, 
        std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& transform)
    {
        std::vector<std::shared_ptr<Material>> materials(1, material);
        addObject(name, shape, materials, hitgroup_prgs, transform);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addObject(const std::string& name, std::shared_ptr<Shape> shape, const std::vector<std::shared_ptr<Material>>& materials, 
        std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& transform)
    {
        ShapeInstance instance{ shape->type(), shape, transform };
        m_objects.emplace_back(Item<Object>{ name, m_current_sbt_id, Object{ shape, materials, instance } });

        // Add hitgroup record data
        for (const auto& m : materials)
        {
            std::array<pgHitgroupRecord, N> hitgroup_records;
            for (uint32_t i = 0; i < N; i++)
                hitgroup_prgs[i].recordPackHeader(&hitgroup_records[i]);
            m_sbt.addHitgroupRecord(hitgroup_records);
        }
        m_current_sbt_id += N * (uint32_t)materials.size();
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::duplicateObject(const std::string& orig_name, const std::string& name, const Matrix4f& transform)
    {
        auto obj = findItem(m_objects, orig_name);
        if (!obj)
        {
            pgLogFatal("The object named with", orig_name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Duplicate object with different transform matrix.
        addObject(name, obj_val.value.shape, obj_val.value.surfaces, transform);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::updateObjectTransform(const std::string& name, const Matrix4f& transform)
    {
        auto obj = findItem(m_objects, name);
        if (!obj)
        {
            pgLogFatal("The object named with", name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Update object's transform matrix.
        obj_val.value.instance.setTransform(transform);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline bool Scene<_CamT, N>::deleteObject(const std::string& name)
    {
        // Search same name object and store the its SBT index
        auto item = deleteItem(m_objects, name);
        if (!item)
            return false;

        auto object = item.value();
        uint32_t deleted_sbt_id = object.ID;
        uint32_t num_surfaces = static_cast<uint32_t>(object.value.surfaces->size());
        uint32_t offset = N * num_surfaces;

        // Offset SBT index in all objects
        for (auto& obj : m_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_moving_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_moving_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }

        return true;
    }

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addLight(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<AreaEmitter> light, 
        std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& transform)
    {
        ShapeInstance instance{ shape->type(), shape, transform };
        m_lights.emplace_back(Item<Light>{ name, m_current_sbt_id, Light{ shape, light, instance } });

        std::array<pgHitgroupRecord, N> hitgroup_records;
        for (uint32_t i = 0; i < N; i++)
            hitgroup_prgs[i].recordPackHeader(&hitgroup_records[i]);
        m_sbt.addHitgroupRecord(hitgroup_records);

        m_current_sbt_id += N;
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::duplicateLight(const std::string& orig_name, const std::string& name, const Matrix4f& transform)
    {
        auto obj = findItem(m_lights, orig_name);
        if (!obj)
        {
            pgLogFatal("The object named with", orig_name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Duplicate object with different transform matrix.
        addLight(name, Light{ obj_val.value.shape(), obj_val.value.emitter, transform });
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::updateLightTransform(const std::string& name, const Matrix4f& transform)
    {
        auto obj = findItem(m_lights, name);
        if (!obj)
        {
            pgLogFatal("The object named with", name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Update object's transform matrix.
        obj_val.value.instance.setTransform(transform);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline bool Scene<_CamT, N>::deleteLight(const std::string& name)
    {
        // Search same name object and store the its SBT index
        auto item = deleteItem(m_lights, name);
        if (!item)
            return false;

        uint32_t deleted_sbt_id = item.value().ID;

        // Offset SBT index in all objects
        for (auto& obj : m_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= N; }
        for (auto& obj : m_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= N; }
        for (auto& obj : m_moving_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= N; }
        for (auto& obj : m_moving_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= N; }

        return true;
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<Material> material, 
        std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        std::vector<std::shared_ptr<Material>> materials(1, material);
        addMovingObject(name, shape, materials, hitgroup_prgs, begin_transform, end_transform, num_key);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingObject(const std::string& name, std::shared_ptr<Shape> shape, const std::vector<std::shared_ptr<Material>>& materials, 
        std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        GeometryAccel gas{ shape->type() };
        gas.addShape(shape);

        // Create transform to reprensents moving object.
        Transform matrix_transform{ TransformType::MatrixMotion };
        matrix_transform.setMatrixMotionTransform(begin_transform, end_transform);
        matrix_transform.setNumKey(num_key);

        m_moving_objects.emplace_back({ name, MovingObject{shape, materials, Instance{}, gas, matrix_transform}, m_current_sbt_id });

        // Add hitgroup record data
        for (const auto& m : materials)
        {
            std::array<pgHitgroupRecord, N> hitgroup_records;
            for (uint32_t i = 0; i < N; i++)
                hitgroup_prgs[i].recordPackHeader(&hitgroup_records[i]);
            m_sbt.addHitgroupRecord(hitgroup_records);
        }
        m_current_sbt_id += N * (uint32_t)materials.size();
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::duplicateMovingObject(const std::string& orig_name, const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        auto obj = findItem(m_moving_objects, orig_name);
        if (!obj)
        {
            pgLogFatal("The object named with", orig_name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Duplicate object with different transform matrix.
        addMovingObject(name, obj_val.value.shape, obj_val.value.surface, begin_transform, end_transform, num_key);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::updateMovingObjectTransform(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform)
    {
        auto obj = findItem(m_moving_objects, name);
        if (!obj)
        {
            pgLogFatal("The object named with", name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Update object's transform matrix.
        obj_val.value.matrix_transform.setMatrixMotionTransform(begin_transform, end_transform);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline bool Scene<_CamT, N>::deleteMovingObject(const std::string& name)
    {
        // Search same name object and store the its SBT index
        auto item = deleteItem(m_moving_objects, name);
        if (!item)
            return false;

        auto object = item.value();
        uint32_t deleted_sbt_id = object.ID;
        uint32_t num_surfaces = static_cast<uint32_t>(object.value.surfaces->size());
        uint32_t offset = N * num_surfaces;

        // Offset SBT index in all objects
        for (auto& obj : m_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_moving_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_moving_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }

        return true;
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingLight(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<AreaEmitter> light, 
        std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        GeometryAccel gas{ shape->type() };
        gas.addShape(shape);

        // Create transform to represents moving object
        Transform matrix_transform{ TransformType::MatrixMotion };
        matrix_transform.setMatrixMotionTransform(begin_transform, end_transform);
        matrix_transform.setNumKey(num_key);

        m_moving_lights.emplace_back({ name, MovingLight{shape, light, Instance{}, gas, matrix_transform}, m_current_sbt_id });

        std::array<pgHitgroupRecord, N> hitgroup_records;
        for (uint32_t i = 0; i < N; i++)
            hitgroup_prgs[i].recordPackHeader(&hitgroup_records[i]);
        m_sbt.addHitgroupRecord(hitgroup_records);
        m_current_sbt_id += N;
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::duplicateMovingLight(const std::string& orig_name, const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        auto obj = findItem(m_moving_lights, orig_name);
        if (!obj)
        {
            pgLogFatal("The object named with", orig_name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Duplicate object with different transform matrix.
        addMovingLight(name, obj.value().shape, obj.value().emitter, begin_transform, end_transform, num_key);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::updateMovingLightTransform(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform)
    {
        auto obj = findItem(m_moving_lights, name);
        if (!obj)
        {
            pgLogFatal("The object named with", name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Update object's transform matrix.
        obj_val.value.matrix_transform.setMatrixMotionTransform(begin_transform, end_transform);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline bool Scene<_CamT, N>::deleteMovingLight(const std::string& name)
    {
        // Search same name object and store the its SBT index
        auto item = deleteItem(m_moving_lights, name);
        if (!item)
            return false;

        uint32_t deleted_sbt_id = item.value().ID;

        // Offset SBT index in all objects
        for (auto& obj : m_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= N; }
        for (auto& obj : m_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= N; }
        for (auto& obj : m_moving_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= N; }
        for (auto& obj : m_moving_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= N; }

        return true;
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline std::vector<std::shared_ptr<AreaEmitter>> Scene<_CamT, N>::areaEmitters() const
    {
        std::vector<std::shared_ptr<AreaEmitter>> area_emitters;

        auto collectEmitters = [&](auto& lights)
        {
            for (auto l : lights)
                area_emitters.push_back(l.value.emitter);
        };

        collectEmitters(m_lights); 
        collectEmitters(m_moving_lights);

        return area_emitters;
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline uint32_t Scene<_CamT, N>::numLights() const
    {
        return static_cast<uint32_t>(m_lights.size() + m_moving_lights.size());
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::copyDataToDevice()
    {
        auto copyObjectDataToDevice = [&](auto& objects)
        {
            for (auto& object : objects)
            {
                auto shape = object.value.shape;
                auto& materials = object.value.materials;

                shape->copyToDevice();
                for (auto& m : materials)
                    m->copyToDevice();
            }
        };

        auto copyLightDataToDevice = [&](auto& lights)
        {
            for (auto& light : lights)
            {
                light.value.shape->copyToDevice();
                light.value.emitter->copyToDevice();
            }
        };

        copyObjectDataToDevice(m_objects);
        copyObjectDataToDevice(m_moving_objects);
        copyLightDataToDevice(m_lights);
        copyLightDataToDevice(m_moving_lights);
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::buildAccel(const Context& ctx, CUstream stream)
    {
        /// @todo : Re-build IAS when it has been already builded.
        
        uint32_t instance_id = 0;
        auto createGas = [&](auto& object, uint32_t ID) -> void
        {
            // Build geometry accel
            object.instance.allowCompaction();
            object.instance.buildAccel(ctx, stream);
            object.instance.setSBTOffset(ID);
            object.instance.setId(instance_id);

            m_accel.addInstance(object.instance);
            instance_id++;
        };

        auto createMovingGas = [&](auto& moving_object, uint32_t ID) -> void
        {
            // Build geometry accel
            moving_object.gas.allowCompaction();
            moving_object.gas.build(ctx, stream);

            // Create transform for moving object
            moving_object.matrix_transform.setChildHandle(moving_object.gas.handle());
            moving_object.matrix_transform.setMotionOptions(m_accel.motionOptions());
            moving_object.matrix_transform.copyToDevice();
            moving_object.matrix_transform.buildHandle(ctx);

            // Set matrix transform to instance
            moving_object.instance.setSBTOffset(ID);
            moving_object.instance.setId(instance_id);
            moving_object.instance.setTraversableHandle(moving_object.matrix_transform.handle());

            m_accel.addInstance(moving_object.instance);

            instance_id++;
        };

        for (auto& obj : m_objects)        createGas(obj.value, obj.ID);
        for (auto& obj : m_lights)         createGas(obj.value, obj.ID);
        for (auto& obj : m_moving_objects) createMovingGas(obj.value, obj.ID);
        for (auto& obj : m_moving_lights)  createMovingGas(obj.value, obj.ID);

        m_accel.build(ctx, stream);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::updateAccel(const Context& ctx, CUstream stream)
    {
        m_accel.update(ctx, stream);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline OptixTraversableHandle Scene<_CamT, N>::accelHandle() const
    {
        return m_accel.handle();
    }

    template <DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::buildSBT()
    {
        // Raygen 
        pgRaygenRecord<_CamT>& rg_record = m_sbt.raygenRecord();
        rg_record.data.camera = m_camera->getData();

        // Miss
        m_envmap->copyToDevice();
        for (uint32_t i = 0; i < N; i++)
        {
            pgMissRecord& ms_record = m_sbt.missRecord(i);
            ms_record.data.env_data = m_envmap->devicePtr();
        }

        auto registerObjectSBTData = [&](auto& objects)
        {
            for (auto& object : objects)
            {
                auto shape = object.value.shape;
                auto& materials = object.value.materials;

                if (!shape->devicePtr())
                    shape->copyToDevice();

                uint32_t ID = object.ID;
                for (auto& m : materials)
                {
                    if (!m->devicePtr())
                        m->copyToDevice();
                    for (uint32_t i = 0; i < N; i++)
                    {
                        pgHitgroupRecord& record = m_sbt.hitgroupRecord(ID + i);
                        record.data = { shape->devicePtr(), m->surfaceInfo() };
                    }
                    ID += N;
                }
            }
        };

        auto registerLightSBTData = [&](auto& lights)
        {
            for (auto& light : lights)
            {
                auto shape = light.value.shape;
                auto emitter = light.value.emitter;

                if (!shape->devicePtr())
                    shape->copyToDevice();
                if (!emitter->devicePtr())
                    emitter->copyToDevice();

                for (uint32_t i = 0; i < N; i++)
                {
                    pgHitgroupRecord& record = m_sbt.hitgroupRecord(light.ID + i);
                    record.data = { shape->devicePtr(), emitter->surfaceInfo() };
                }
            }
        };

        registerObjectSBTData(m_objects);
        registerObjectSBTData(m_moving_objects);
        registerLightSBTData(m_lights);
        registerLightSBTData(m_moving_lights);

        // Build SBT on device
        m_sbt.createOnDevice();
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::updateSBT(uint32_t record_type)
    {
        if (!m_sbt.isOnDevice())
        {
            pgLogWarn("Shader binding table has not been allocated on device yet.");
            return;
        }

        // Update raygen data on device
        if (+(record_type & SBTRecordType::Raygen))
        {
            // Get device pointer to the SBT record
            auto* rg_record = reinterpret_cast<pgRaygenRecord<_CamT>*>(m_sbt.deviceRaygenRecordPtr());
            pgRaygenData<_CamT> rg_data;
            rg_data.camera = m_camera->getData();

            // Upload camera data to device
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(&rg_record->data),
                &rg_data, sizeof(pgRaygenData<_CamT>),
                cudaMemcpyHostToDevice
            ));
        }

        // Update miss data on device
        if (+(record_type & SBTRecordType::Miss))
        {
            // Get device pointer to the SBT record
            auto* ms_record = reinterpret_cast<pgMissRecord*>(m_sbt.deviceMissRecordPtr());

            pgMissData ms_data;
            ms_data.env_data = m_envmap->devicePtr();
            
            // Upload envmap data to device for each ray types
            for (uint32_t i = 0; i < N; i++)
            {
                CUDA_CHECK(cudaMemcpy(
                    reinterpret_cast<void*>(&ms_record[i].data),
                    &ms_data, sizeof(pgMissData),
                    cudaMemcpyHostToDevice
                ));
            }
        }

        // Update hitgroup data on device
        if (+(record_type & SBTRecordType::Hitgroup))
        {
            std::vector<pgHitgroupData> hitgroup_datas;
            auto collectObjectSBTData = [&](auto& objects)
            {
                for (auto& object : objects)
                {
                    auto shape = object.value.shape;
                    auto& materials = object.value.materials;

                    shape->copyToDevice();

                    // Set hitgroup data to records
                    for (auto& m : materials) {
                        m->copyToDevice();
                        pgHitgroupData hg_data;
                        hg_data = { shape->devicePtr(), m->surfaceInfo() };
                        for (uint32_t i = 0; i < N; i++)
                            hitgroup_datas.emplace_back(hg_data);
                    }
                }
            };

            auto collectLightSBTData = [&](auto& lights)
            {
                for (auto& light : lights)
                {
                    light.value.shape->copyToDevice();
                    light.value.emitter->copyToDevice();
                    pgHitgroupData hg_data;
                    hg_data = { light.value.shape->devicePtr(), light.value.emitter->devicePtr() };
                    for (uint32_t i = 0; i < N; i++)
                        hitgroup_datas.emplace_back(hg_data);
                }
            };

            collectObjectSBTData(m_objects);
            collectObjectSBTData(m_moving_objects);
            collectLightSBTData(m_lights);
            collectLightSBTData(m_moving_lights);
            
            /* Copy hitgroup data with the header-part stride.
             * "+ OPTIX_SBT_RECORD_HEADER_SIZE" is for making dst a pointer to the 0th hitgroup data.
             * 
             * Array of pgHitgroupRecord ...
             *                 | pgHitgroupRecord        | pgHitgroupRecord        | ...
             * record inside : | header | pgHitgroupData | header | pgHitgroupData | ...
             *                            �� "+ OPTIX_SBT_RECORD_HEADER_SIZE" makes 'dst' start here */
            CUDA_CHECK(cudaMemcpy2D(
                /* dst = */ reinterpret_cast<void*>(m_sbt.deviceHitgroupRecordPtr() + OPTIX_SBT_RECORD_HEADER_SIZE), /* dpitch = */ sizeof(pgHitgroupRecord),
                /* src = */ hitgroup_datas.data(), /* spitch = */ sizeof(pgHitgroupData),
                /* width = */ sizeof(pgHitgroupData), /* height = */ hitgroup_datas.size(), 
                /* kind = */ cudaMemcpyHostToDevice
            ));
        }
    }

} // namespace prayground