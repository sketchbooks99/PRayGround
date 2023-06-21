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

    template <DerivedFromCamera _CamT, uint32_t _NRay>
    class Scene {
    // Internal classes
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
            std::vector<std::shared_ptr<AreaEmitter>> emitters;
            ShapeInstance instance;
        };

        struct MovingLight {
            std::shared_ptr<Shape> shape;
            std::vector<std::shared_ptr<AreaEmitter>> emitters;

            Instance instance;
            GeometryAccel gas;
            Transform matrix_transform;
        };

    // Public interfaces
    public:
        static constexpr uint32_t NRay = _NRay;
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
        void bindMissPrograms(std::array<ProgramGroup, _NRay>& miss_prgs);
        void bindCallablesProgram(ProgramGroup& prg);
        void bindExceptionProgram(ProgramGroup& prg);

        void setCamera(const std::shared_ptr<_CamT>& camera);
        const std::shared_ptr<_CamT>& camera();

        // Automatically load and create envmap texture from file
        void setEnvmap(const std::shared_ptr<Texture>& texture);
        std::shared_ptr<EnvironmentEmitter> envmap() const;

        /// @note Should create/deletion functions for object return boolean value?
        // Object
        void addObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<Material> material, 
            std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& transform = Matrix4f::identity());
        void addObject(const std::string& name, std::shared_ptr<Shape> shape, const std::vector<std::shared_ptr<Material>>& materials,
            std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& transform = Matrix4f::identity());
        void duplicateObject(const std::string& orig_name, const std::string& name, const Matrix4f& transform = Matrix4f::identity());
        void updateObjectTransform(const std::string& name, const Matrix4f& transform);
        bool deleteObject(const std::string& name);

        // Light object
        void addLight(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<AreaEmitter> emitter,
            std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& transform = Matrix4f::identity());
        void addLight(const std::string& name, std::shared_ptr<Shape> shape, const std::vector<std::shared_ptr<AreaEmitter>>& emitters,
            std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& transform = Matrix4f::identity());
        void duplicateLight(const std::string& orig_name, const std::string& name, const Matrix4f& transform = Matrix4f::identity());
        void updateLightTransform(const std::string& name, const Matrix4f& transform);
        bool deleteLight(const std::string& name);

        // Moving object (especially for motion blur)
        void addMovingObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<Material> material,
            std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void addMovingObject(const std::string& name, std::shared_ptr<Shape> shape, const std::vector<std::shared_ptr<Material>>& materials,
            std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void duplicateMovingObject(const std::string& orig_name, const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void updateMovingObjectTransform(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform);
        bool deleteMovingObject(const std::string& name);

        // Moving light (especially for motion blur)
        void addMovingLight(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<AreaEmitter> emitter,
            std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void addMovingLight(const std::string& name, std::shared_ptr<Shape> shape, const std::vector<std::shared_ptr<AreaEmitter>>& emitters,
            std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
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
        uint32_t                        m_num_lights;

        // Flag represents scene states should be updated.
        bool should_accel_updated;
        bool should_sbt_updated;
    };

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t _NRay>
    inline Scene<_CamT, _NRay>::Scene()
    {
        this->setup();
    }

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::setup()
    {
        Settings settings;
        settings.allow_motion           = false;
        settings.allow_accel_compaction = true;
        settings.allow_accel_update     = false;
        this->setup(settings);
    }

    template <DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::setup(const Scene::Settings& settings)
    {
        m_settings = settings;

        // Initialize instance acceleration structure
        m_accel = InstanceAccel{ InstanceAccel::Type::Instances };

        // Initialize raygen and miss record
        pgRaygenRecord<_CamT> rg_record = {};
        m_sbt.setRaygenRecord(rg_record);
        
        std::array<pgMissRecord, _NRay> ms_records{};
        m_sbt.setMissRecord(ms_records);
    }

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t _NRay>
    template <class LaunchParams>
    inline void Scene<_CamT, _NRay>::launchRay(
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
    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::bindRaygenProgram(ProgramGroup& rg_prg)
    {
        // Fill the record header with the raygen program
        pgRaygenRecord<_CamT>& rg_record = m_sbt.raygenRecord();
        rg_prg.recordPackHeader(&rg_record);
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::bindMissPrograms(std::array<ProgramGroup, _NRay>& miss_prgs)
    {
        // Fill record's headers with the miss programs
        for (uint32_t i = 0; i < _NRay; i++)
        {
            pgMissRecord& record = m_sbt.missRecord(i);
            miss_prgs[i].recordPackHeader(&record);
        }
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::bindCallablesProgram(ProgramGroup& prg)
    {
        // Add SBT record and fill the record header at the same time
        // since the pgCallablesRecord has no data
        pgCallablesRecord record{};
        prg.recordPackHeader(&record);
        m_sbt.addCallablesRecord(record);
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::bindExceptionProgram(ProgramGroup& prg)
    {
        // Add SBT record and fill the record header at the same time
        // since the pgExceptionRecord has no data
        pgExceptionRecord record{};
        prg.recordPackHeader(&record);
        m_sbt.setExceptionRecord(record);
    }

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::setCamera(const std::shared_ptr<_CamT>& camera)
    {
        m_camera = camera;
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline const std::shared_ptr<_CamT>& Scene<_CamT, _NRay>::camera()
    {
        return m_camera;
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::setEnvmap(const std::shared_ptr<Texture>& texture)
    {
        if (!m_envmap) 
            m_envmap = std::make_shared<EnvironmentEmitter>(texture);
        else
            m_envmap->setTexture(texture);
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline std::shared_ptr<EnvironmentEmitter> Scene<_CamT, _NRay>::envmap() const
    {
        return m_envmap;
    }

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::addObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<Material> material, 
        std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& transform)
    {
        std::vector<std::shared_ptr<Material>> materials(1, material);
        addObject(name, shape, materials, hitgroup_prgs, transform);
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::addObject(const std::string& name, std::shared_ptr<Shape> shape, const std::vector<std::shared_ptr<Material>>& materials, 
        std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& transform)
    {
        ShapeInstance instance{ shape->type(), shape, transform };
        m_objects.emplace_back(Item<Object>{ name, m_current_sbt_id, Object{ shape, materials, instance } });

        // Add hitgroup record data
        for ([[maybe_unused]] const auto& m : materials)
        {
            std::array<pgHitgroupRecord, _NRay> hitgroup_records;
            for (uint32_t i = 0; i < _NRay; i++)
                hitgroup_prgs[i].recordPackHeader(&hitgroup_records[i]);
            m_sbt.addHitgroupRecord(hitgroup_records);
        }
        m_current_sbt_id += _NRay * (uint32_t)materials.size();
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::duplicateObject(const std::string& orig_name, const std::string& name, const Matrix4f& transform)
    {
        auto obj = findItem(m_objects, orig_name);
        if (!obj)
        {
            pgLogFatal("The object named with", orig_name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Duplicate object with different transform matrix.
        addObject(name, obj_val.value.shape, obj_val.value.materials, transform);
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::updateObjectTransform(const std::string& name, const Matrix4f& transform)
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

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline bool Scene<_CamT, _NRay>::deleteObject(const std::string& name)
    {
        // Search same name object and store the its SBT index
        auto item = deleteItem(m_objects, name);
        if (!item)
            return false;

        auto object = item.value();
        uint32_t deleted_sbt_id = object.ID;
        uint32_t num_materials = static_cast<uint32_t>(object.value.materials.size());
        uint32_t offset = _NRay * num_materials;

        // Offset SBT index in all objects
        for (auto& obj : m_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_moving_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_moving_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }

        /** 
         * @todo Should the SBT record corresponds to object be also deleted? 
         * If so, I have to use m_sbt.deleteHitgroupRecord(idx) for deleted object
         */ 

        return true;
    }

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::addLight(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<AreaEmitter> emitter, 
        std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& transform)
    {
        std::vector<std::shared_ptr<AreaEmitter>> emitters(1, emitter);
        addLight(name, shape, emitters, hitgroup_prgs, transform);
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::addLight(const std::string& name, std::shared_ptr<Shape> shape, const std::vector<std::shared_ptr<AreaEmitter>>& emitters, 
        std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& transform)
    {
        ShapeInstance instance{ shape->type(), shape, transform };
        m_lights.emplace_back(Item<Light>{ name, m_current_sbt_id, Light{ shape, emitters, instance } });

        // Add hitgroup record data
        for ([[maybe_unused]] const auto& e : emitters)
        {
            std::array<pgHitgroupRecord, _NRay> hitgroup_records;
            for (uint32_t i = 0; i < _NRay; i++)
                hitgroup_prgs[i].recordPackHeader(&hitgroup_records[i]);
            m_sbt.addHitgroupRecord(hitgroup_records);
        }
        m_current_sbt_id += _NRay * (uint32_t)emitters.size();
        m_num_lights += shape->numPrimitives();
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::duplicateLight(const std::string& orig_name, const std::string& name, const Matrix4f& transform)
    {
        auto obj = findItem(m_lights, orig_name);
        if (!obj)
        {
            PG_LOG_FATAL("The object named with", orig_name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Duplicate object with different transform matrix.
        addLight(name, Light{ obj_val.value.shape, obj_val.value.emitters, transform });
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::updateLightTransform(const std::string& name, const Matrix4f& transform)
    {
        auto obj = findItem(m_lights, name);
        if (!obj)
        {
            PG_LOG_FATAL("The object named with", name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Update object's transform matrix.
        obj_val.value.instance.setTransform(transform);
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline bool Scene<_CamT, _NRay>::deleteLight(const std::string& name)
    {
        // Search same name object and store the its SBT index
        auto item = deleteItem(m_lights, name);
        if (!item)
            return false;

        auto light = item.value();

        uint32_t deleted_sbt_id = light.ID;
        uint32_t num_emitters = static_cast<uint32_t>(light.value.emitters.size());
        uint32_t offset = _NRay * num_emitters;

        m_num_lights -= light.value.shape->numPrimitives();

        // Offset SBT index in all objects
        for (auto& obj : m_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_moving_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_moving_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }

        return true;
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::addMovingObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<Material> material, 
        std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        std::vector<std::shared_ptr<Material>> materials(1, material);
        addMovingObject(name, shape, materials, hitgroup_prgs, begin_transform, end_transform, num_key);
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::addMovingObject(const std::string& name, std::shared_ptr<Shape> shape, const std::vector<std::shared_ptr<Material>>& materials, 
        std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        GeometryAccel gas{ shape->type() };
        gas.addShape(shape);

        // Create transform to reprensents moving object.
        Transform matrix_transform{ TransformType::MatrixMotion };
        matrix_transform.setMatrixMotionTransform(begin_transform, end_transform);
        matrix_transform.setNumKey(num_key);

        m_moving_objects.emplace_back(Item<MovingObject>{ name, m_current_sbt_id, MovingObject{shape, materials, Instance{}, gas, matrix_transform} });

        // Add hitgroup record data
        for (const auto& m : materials)
        {
            std::array<pgHitgroupRecord, _NRay> hitgroup_records;
            for (uint32_t i = 0; i < _NRay; i++)
                hitgroup_prgs[i].recordPackHeader(&hitgroup_records[i]);
            m_sbt.addHitgroupRecord(hitgroup_records);
        }
        m_current_sbt_id += _NRay * (uint32_t)materials.size();
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::duplicateMovingObject(const std::string& orig_name, const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        auto obj = findItem(m_moving_objects, orig_name);
        if (!obj)
        {
            pgLogFatal("The object named with", orig_name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Duplicate object with different transform matrix.
        addMovingObject(name, obj_val.value.shape, obj_val.value.materials, begin_transform, end_transform, num_key);
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::updateMovingObjectTransform(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform)
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

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline bool Scene<_CamT, _NRay>::deleteMovingObject(const std::string& name)
    {
        // Search same name object and store the its SBT index
        auto item = deleteItem(m_moving_objects, name);
        if (!item)
            return false;

        auto object = item.value();
        uint32_t deleted_sbt_id = object.ID;
        uint32_t num_materials = static_cast<uint32_t>(object.value.materials.size());
        uint32_t offset = _NRay * num_materials;

        // Offset SBT index in all objects
        for (auto& obj : m_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_moving_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_moving_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }

        return true;
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::addMovingLight(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<AreaEmitter> emitter, 
        std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        std::vector<std::shared_ptr<AreaEmitter>> emitters(1, emitter);
        addMovingLight(name, shape, emitters, hitgroup_prgs, begin_transform, end_transform, num_key);
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::addMovingLight(const std::string& name, std::shared_ptr<Shape> shape, const std::vector<std::shared_ptr<AreaEmitter>>& emitters, 
        std::array<ProgramGroup, _NRay>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        GeometryAccel gas{ shape->type() };
        gas.addShape(shape);

        // Create transform to reprensents moving object.
        Transform matrix_transform{ TransformType::MatrixMotion };
        matrix_transform.setMatrixMotionTransform(begin_transform, end_transform);
        matrix_transform.setNumKey(num_key);

        m_moving_lights.emplace_back(Item<MovingLight>{ name, m_current_sbt_id, MovingLight{shape, emitters, Instance{}, gas, matrix_transform} });

        // Add hitgroup record data
        for (const auto& e : emitters)
        {
            std::array<pgHitgroupRecord, _NRay> hitgroup_records;
            for (uint32_t i = 0; i < _NRay; i++)
                hitgroup_prgs[i].recordPackHeader(&hitgroup_records[i]);
            m_sbt.addHitgroupRecord(hitgroup_records);
        }
        m_current_sbt_id += _NRay * (uint32_t)emitters.size();
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::duplicateMovingLight(const std::string& orig_name, const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        auto obj = findItem(m_moving_lights, orig_name);
        if (!obj)
        {
            pgLogFatal("The object named with", orig_name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Duplicate object with different transform matrix.
        addMovingLight(name, obj.value().shape, obj.value().emitters, begin_transform, end_transform, num_key);
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::updateMovingLightTransform(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform)
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

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline bool Scene<_CamT, _NRay>::deleteMovingLight(const std::string& name)
    {
        // Search same name object and store the its SBT index
        auto item = deleteItem(m_moving_lights, name);
        if (!item)
            return false;

        auto light = item.value();
        uint32_t deleted_sbt_id = light.ID;
        uint32_t num_emitters = static_cast<uint32_t>(light.value.emitters.size());
        uint32_t offset = _NRay * num_emitters;

        m_num_lights -= light.value.shape->numPrimitives();

        // Offset SBT index in all objects
        for (auto& obj : m_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_moving_objects) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }
        for (auto& obj : m_moving_lights) { if (obj.ID > deleted_sbt_id) obj.ID -= offset; }

        return true;
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline std::vector<std::shared_ptr<AreaEmitter>> Scene<_CamT, _NRay>::areaEmitters() const
    {
        std::vector<std::shared_ptr<AreaEmitter>> area_emitters;

        auto collectEmitters = [&](auto& lights)
        {
            for (auto& l : lights)
                std::copy(l.value.emitters.begin(), l.value.emitters.end(), std::back_inserter(area_emitters));
        };

        collectEmitters(m_lights); 
        collectEmitters(m_moving_lights);

        return area_emitters;
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline uint32_t Scene<_CamT, _NRay>::numLights() const
    {
        return m_num_lights;
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::copyDataToDevice()
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
                for (auto& e : light.value.emitters)
                    e->copyToDevice();
            }
        };

        copyObjectDataToDevice(m_objects);
        copyObjectDataToDevice(m_moving_objects);
        copyLightDataToDevice(m_lights);
        copyLightDataToDevice(m_moving_lights);
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::buildAccel(const Context& ctx, CUstream stream)
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

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::updateAccel(const Context& ctx, CUstream stream)
    {
        m_accel.update(ctx, stream);
    }

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline OptixTraversableHandle Scene<_CamT, _NRay>::accelHandle() const
    {
        return m_accel.handle();
    }

    template <DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::buildSBT()
    {
        // Raygen 
        pgRaygenRecord<_CamT>& rg_record = m_sbt.raygenRecord();
        rg_record.data.camera = m_camera->getData();

        // Miss
        m_envmap->copyToDevice();
        for (uint32_t i = 0; i < _NRay; i++)
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
                    for (uint32_t i = 0; i < _NRay; i++)
                    {
                        pgHitgroupRecord& record = m_sbt.hitgroupRecord(ID + i);
                        record.data = { shape->devicePtr(), m->surfaceInfo() };
                    }
                    ID += _NRay;
                }
            }
        };

        auto registerLightSBTData = [&](auto& lights)
        {
            for (auto& light : lights)
            {
                auto shape = light.value.shape;
                auto& emitters = light.value.emitters;

                if (!shape->devicePtr())
                    shape->copyToDevice();

                uint32_t ID = light.ID;
                for (auto& e : emitters)
                {
                    if (!e->devicePtr())
                        e->copyToDevice();
                    for (uint32_t i = 0; i < _NRay; i++)
                    {
                        pgHitgroupRecord& record = m_sbt.hitgroupRecord(ID + i);
                        record.data = { shape->devicePtr(), e->surfaceInfo() };
                    }
                    ID += _NRay;
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

    template<DerivedFromCamera _CamT, uint32_t _NRay>
    inline void Scene<_CamT, _NRay>::updateSBT(uint32_t record_type)
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

            m_envmap->copyToDevice();
            pgMissData ms_data;
            ms_data.env_data = m_envmap->devicePtr();
            
            // Upload envmap data to device for each ray types
            for (uint32_t i = 0; i < _NRay; i++)
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
                        for (uint32_t i = 0; i < _NRay; i++)
                            hitgroup_datas.emplace_back(hg_data);
                    }
                }
            };

            auto collectLightSBTData = [&](auto& lights)
            {
                for (auto& light : lights)
                {
                    auto shape = light.value.shape;
                    auto& emitters = light.value.emitters;

                     shape->copyToDevice();

                    // Set hitgroup data to records
                    for (auto& e : emitters) {
                        e->copyToDevice();
                        pgHitgroupData hg_data;
                        hg_data = { shape->devicePtr(), e->surfaceInfo() };
                        for (uint32_t i = 0; i < _NRay; i++)
                            hitgroup_datas.emplace_back(hg_data);
                    }
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
             *                            ^ "+ OPTIX_SBT_RECORD_HEADER_SIZE" makes 'dst' start here */
            CUDA_CHECK(cudaMemcpy2D(
                /* dst = */ reinterpret_cast<void*>(m_sbt.deviceHitgroupRecordPtr() + OPTIX_SBT_RECORD_HEADER_SIZE), /* dpitch = */ sizeof(pgHitgroupRecord),
                /* src = */ hitgroup_datas.data(), /* spitch = */ sizeof(pgHitgroupData),
                /* width = */ sizeof(pgHitgroupData), /* height = */ hitgroup_datas.size(), 
                /* kind = */ cudaMemcpyHostToDevice
            ));
        }
    }

} // namespace prayground