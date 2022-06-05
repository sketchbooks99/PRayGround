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
    template <class T>
    concept DerivedFromCamera = std::derived_from<T, Camera>;

    template <DerivedFromCamera _CamT, uint32_t N>
    class Scene {
    private:
        // ID is for shader binding table offset
        template <class T>
        struct Item {
            std::string name;
            uint32_t ID;
            T value;
        };

        template <class SurfaceT>
        struct Object_ {
            std::shared_ptr<Shape> shape;
            std::shared_ptr<SurfaceT> surface;
            ShapeInstance instance;
        };

        template <class SurfaceT>
        struct MovingObject_ {
            std::shared_ptr<Shape> shape;
            std::shared_ptr<SurfaceT> surface;

            Instance instance;
            GeometryAccel gas;
            Transform matrix_transform;
        };
    public:
        static constexpr uint32_t NRay = N;
        using CamT = _CamT;
        using SBT = pgDefaultSBT<CamT, NRay>;

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
        void duplicateObject(const std::string& orig_name, const std::string& name, const Matrix4f& transform = Matrix4f::identity());
        void updateObjectTransform(const std::string& name, const Matrix4f& transform);

        // Light object
        void addLightObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<AreaEmitter> light,
            std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& transform = Matrix4f::identity());
        void duplicateLightObject(const std::string& orig_name, const std::string& name, const Matrix4f& transform = Matrix4f::identity());
        void updateLightObjectTransform(const std::string& name, const Matrix4f& transform);

        // Moving object (especially for motion blur)
        void addMovingObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<Material> material,
            std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void duplicateMovingObject(const std::string& orig_name, const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void updateMovingObjectTransform(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform);

        // Moving light (especially for motion blur)
        void addMovingLightObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<AreaEmitter> light,
            std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void duplicateMovingLightObject(const std::string& orig_name, const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key = 2);
        void updateMovingLightObjectTransform(const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform);

        // Erase object and corresponding shader binding table record
        void deleteObject(const std::string& name);

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
        std::vector<Item<LightObject>>        m_light_objects;
        std::vector<Item<MovingLightObject>>  m_moving_light_objects;
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

        m_accel = InstanceAccel{ InstanceAccel::Type::Instances };

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

        optixLaunch(
            static_cast<OptixPipeline>(ppl), stream,
            d_params.devicePtr(), d_params.size(), &m_sbt.sbt(),
            w, h, d);
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::bindRaygenProgram(ProgramGroup& rg_prg)
    {
        pgRaygenRecord<_CamT>& rg_record = m_sbt.raygenRecord();
        rg_prg.recordPackHeader(&rg_record);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::bindMissPrograms(std::array<ProgramGroup, N>& miss_prgs)
    {
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
        pgCallablesRecord record{};
        prg.recordPackHeader(&record);
        m_sbt.addCallablesRecord(record);
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::bindExceptionProgram(ProgramGroup& prg)
    {
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
        return camera;
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
        ShapeInstance instance{ shape->type(), shape, transform };
        m_objects.emplace_back(Item<Object>{ name, m_current_sbt_id, Object{shape, material, instance} });
        std::array<pgHitgroupRecord, N> hitgroup_records;
        for (uint32_t i = 0; i < N; i++)
            hitgroup_prgs[i].recordPackHeader(&hitgroup_records[i]);
        m_sbt.addHitgroupRecord(hitgroup_records);
        m_current_sbt_id += N;
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
        addObject(name, obj_val.value.shape, obj_val.value.surface, transform);
    }

    // -------------------------------------------------------------------------------
    template <DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addLightObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<AreaEmitter> area, 
        std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& transform)
    {
        ShapeInstance instance{ shape->type(), shape, transform };
        m_light_objects.emplace_back(Item<LightObject>{ name, m_current_sbt_id, LightObject{shape, area, instance} });
        std::array<pgHitgroupRecord, N> hitgroup_records;
        for (uint32_t i = 0; i < N; i++)
            hitgroup_prgs[i].recordPackHeader(&hitgroup_records[i]);
        m_sbt.addHitgroupRecord(hitgroup_records);
        m_current_sbt_id += N;
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::duplicateLightObject(const std::string& orig_name, const std::string& name, const Matrix4f& transform)
    {
        auto obj = findItem(m_light_objects, orig_name);
        if (!obj)
        {
            pgLogFatal("The object named with", orig_name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Duplicate object with different transform matrix.
        addLightObject(name, Object{ obj_val.value.shape(), obj_val.value.surface, transform });
    }

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<Material> material, 
        std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        GeometryAccel gas{shape->type()};
        gas.addShape(shape);

        Transform matrix_transform{ TransformType::MatrixMotion }; 
        matrix_transform.setMatrixMotionTransform(begin_transform, end_transform);
        matrix_transform.setNumKey(num_key);

        m_moving_objects.emplace_back({ name, MovingObject{shape, material, Instance{}, gas, matrix_transform}, m_current_sbt_id });
        std::array<pgHitgroupRecord, N> hitgroup_records;
        for (uint32_t i = 0; i < N; i++)
            hitgroup_prgs[i].recordPackHeader(&hitgroup_records[i]);
        m_sbt.addHitgroupRecord(hitgroup_records);
        m_current_sbt_id += N;
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

    // -------------------------------------------------------------------------------
    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::addMovingLightObject(const std::string& name, std::shared_ptr<Shape> shape, std::shared_ptr<AreaEmitter> light, 
        std::array<ProgramGroup, N>& hitgroup_prgs, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        GeometryAccel gas{ shape->type() };
        gas.addShape(shape);

        Transform matrix_transform{ TransformType::MatrixMotion };
        matrix_transform.setMatrixMotionTransform(begin_transform, end_transform);
        matrix_transform.setNumKey(num_key);

        m_moving_light_objects.emplace_back({ name, MovingLightObject{shape, light, Instance{}, gas, matrix_transform}, m_current_sbt_id });
        std::array<pgHitgroupRecord, N> hitgroup_records;
        for (uint32_t i = 0; i < N; i++)
            hitgroup_prgs[i].recordPackHeader(&hitgroup_records[i]);
        m_sbt.addHitgroupRecord(hitgroup_records);
        m_current_sbt_id += N;
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::duplicateMovingLightObject(const std::string& orig_name, const std::string& name, const Matrix4f& begin_transform, const Matrix4f& end_transform, uint16_t num_key)
    {
        auto obj = findItem(m_moving_light_objects, orig_name);
        if (!obj)
        {
            pgLogFatal("The object named with", orig_name, "is not found.");
            return;
        }

        auto& obj_val = obj.value();

        // Duplicate object with different transform matrix.
        addMovingLightObject(name, obj.value().shape, obj.value().surface(), begin_transform, end_transform, num_key);
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::deleteObject(const std::string& name)
    {
        // Search same name object and store the its SBT index
        uint32_t deleted_sbt_id;
        auto deleteObj = [&](const auto& objects)
        {
            for (auto it = objects.begin(); it != objects.end();)
            {
                if (it->name == name)
                {
                    deleted_sbt_id = it->ID;
                    objects.erase(it);
                    return true;
                }
            }
            return false;
        };

        // Offset SBT index in all objects 
        auto offsetSBTIndex = [&]()
        {
            for (auto& obj : m_objects)                 { if (obj.ID > deleted_sbt_id) obj.ID -= N; }
            for (auto& obj : m_light_objects)           { if (obj.ID > deleted_sbt_id) obj.ID -= N; }
            for (auto& obj : m_moving_objects)          { if (obj.ID > deleted_sbt_id) obj.ID -= N; }
            for (auto& obj : m_moving_light_objects)    { if (obj.ID > deleted_sbt_id) obj.ID -= N; }
        };

        if (deleteObj(m_objects))               offsetSBTIndex();
        if (deleteObj(m_light_objects))         offsetSBTIndex();
        if (deleteObj(m_moving_objects))        offsetSBTIndex();
        if (deleteObj(m_moving_light_objects))  offsetSBTIndex();
    }

    template<DerivedFromCamera _CamT, uint32_t N>
    inline void Scene<_CamT, N>::buildAccel(const Context& ctx, CUstream stream)
    {
        /// @todo : Re-build IAS when it has been already builded.
        
        uint32_t instance_id = 0;
        auto createGas = [&](auto& object, uint32_t ID) -> void
        {
            // Set shader binding table index to shape
            //object.shape->setSbtIndex(ID);

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
            // Set shader binding table index to shape 
            //moving_object.shape->setSbtIndex(ID);

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

        for (auto& obj : m_objects)              createGas(obj.value, obj.ID);
        for (auto& obj : m_light_objects)        createGas(obj.value, obj.ID);
        for (auto& obj : m_moving_objects)       createMovingGas(obj.value, obj.ID);
        for (auto& obj : m_moving_light_objects) createMovingGas(obj.value, obj.ID);

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

        // Hitgroup
        auto registerSBTData = [&](auto& objects)
        {
            for (auto& obj : m_objects)
            {
                auto& shape = obj.value.shape;
                auto& surface = obj.value.surface;

                shape->copyToDevice();
                shape->setSbtIndex(obj.ID);
                surface->copyToDevice();

                // Set actual sbt data to records
                for (int i = 0; i < N; i++)
                {
                    pgHitgroupRecord& record = m_sbt.hitgroupRecord(obj.ID + i);
                    record.data =
                    {
                        .shape_data = shape->devicePtr(),
                        .surface_info =
                        {
                            .data = surface->devicePtr(),
                            .callable_id = {
                                .sample = surface->surfaceCallableID().sample,
                                .bsdf = surface->surfaceCallableID().bsdf,
                                .pdf = surface->surfaceCallableID().pdf
                            },
                            .type = surface->surfaceType()
                        }
                    };
                }
            }
        };

        registerSBTData(m_objects);
        registerSBTData(m_light_objects);
        registerSBTData(m_moving_objects);
        registerSBTData(m_moving_light_objects);

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

        if (+(record_type & SBTRecordType::Raygen))
        {
            auto* rg_record = reinterpret_cast<pgRaygenRecord<_CamT>*>(m_sbt.deviceRaygenRecordPtr());
            pgRaygenData<_CamT> rg_data;
            rg_data.camera = m_camera->getData();
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(&rg_record->data),
                &rg_data, sizeof(pgRaygenData<_CamT>),
                cudaMemcpyHostToDevice
            ));
        }
    }

} // namespace prayground