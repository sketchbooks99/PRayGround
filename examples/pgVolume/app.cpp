#include "app.h"

void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.devicePtr());
}

void App::handleCameraUpdate()
{
    if (!camera_update)
        return;
    camera_update = false;

    RaygenRecord* rg_record = reinterpret_cast<RaygenRecord*>(sbt.raygenRecord());
    RaygenData rg_data = { .camera = camera.getData() };
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(&rg_record->data),
        &rg_data, sizeof(RaygenData),
        cudaMemcpyHostToDevice
    ));

    initResultBufferOnDevice();
}

// ------------------------------------------------------------------
void App::setup()
{
    // Initialize CUDA
    stream = 0;
    CUDA_CHECK(cudaFree(0));
    // Initialize OptixDeviceContext
    OPTIX_CHECK(optixInit());
    context.disableValidation();
    context.create();

    // Initialize instance acceleration structure
    ias = InstanceAccel{ InstanceAccel::Type::Instances };

    // Pipeline settings
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(5);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);

    // Create module from CUDA code
    Module module = pipeline.createModuleFromCudaFile(context, "kernels.cu");

    // Initialize bitmaps to store rendered results
    result_bmp.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());
    accum_bmp.allocate(PixelFormat::RGBA, pgGetWidth(), pgGetHeight());

    // Configuration of launch parameters
    params.width = result_bmp.width();
    params.height = result_bmp.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;

    initResultBufferOnDevice();

    // Camera settings
    camera.setOrigin(0, 0, -1000);
    camera.setLookat(0, 0, 0);
    camera.setUp(0, 1, 0);
    camera.setFarClip(5000);
    camera.setFov(40);
    camera.setAspect(static_cast<float>(params.width) / params.height);
    camera.enableTracking(pgGetCurrentWindow());

    // Raygen program
    ProgramGroup rg_prg = pipeline.createRaygenProgram(context, module, "__raygen__medium");
    // Shader binding table data for raygen program
    RaygenRecord rg_record;
    rg_prg.recordPackHeader(&rg_record);
    rg_record.data.camera = camera.getData();
    sbt.setRaygenRecord(rg_record);

    auto setupCallable = [&](const std::string& dc, const std::string& cc)
        -> uint32_t
    {
        EmptyRecord ca_record = {};
        auto [prg, id] = pipeline.createCallablesProgram(context, module, dc, cc);
        prg.recordPackHeader(&ca_record);
        sbt.addCallablesRecord(ca_record);
        return id;
    };

    // Callable programs for textures
    uint32_t constant_prg_id = setupCallable(DC_FUNC_STR("constant"), "");
    uint32_t checker_prg_id = setupCallable(DC_FUNC_STR("checker"), "");
    uint32_t bitmap_prg_id = setupCallable(DC_FUNC_STR("bitmap"), "");

    // Callable programs for materials
    // Diffuse
    uint32_t diffuse_prg_id = setupCallable(DC_FUNC_STR("sample_diffuse"), "");
    // Medium
    uint32_t medium_prg_id = setupCallable(DC_FUNC_STR("sample_medium"), "");
    // Area emitter
    uint32_t area_emitter_prg_id = setupCallable(DC_FUNC_STR("area_emitter"), "");
    
    textures.emplace("env", new FloatBitmapTexture("resources/image/sepulchral_chapel_rotunda_4k.exr", bitmap_prg_id));

    env = EnvironmentEmitter{ textures.at("env") };
    env.copyToDevice();

    // Miss program
    ProgramGroup ms_prg = pipeline.createMissProgram(context, module, "__miss__envmap");
    // Shader binding table data for miss program
    MissRecord ms_record;
    ms_prg.recordPackHeader(&ms_record);
    ms_record.data.env_data = env.devicePtr();
    sbt.setMissRecord(ms_record);

    // Hitgroup programs
    // Plane
    auto plane_prg = pipeline.createHitgroupProgram(context, module, CH_FUNC_STR("plane"), IS_FUNC_STR("plane"));
    // Grid medium
    auto grid_prg = pipeline.createHitgroupProgram(context, module, CH_FUNC_STR("grid"), IS_FUNC_STR("grid"));

    struct Primitive
    {
        shared_ptr<Shape> shape;
        shared_ptr<Material> material;
        uint32_t sample_id;
    };

    uint32_t sbt_idx = 0;
    uint32_t sbt_offset = 0;
    uint32_t instance_id = 0;

    using SurfaceP = variant<shared_ptr<Material>, shared_ptr<AreaEmitter>>;
    auto addHitgroupRecord = [&](ProgramGroup& prg, shared_ptr<Shape> shape, SurfaceP surface, uint32_t sample_id)
    {
        const bool is_mat = holds_alternative<shared_ptr<Material>>(surface);

        // Copy data to GPU
        shape->copyToDevice();
        shape->setSbtIndex(sbt_idx);
        if (is_mat)
            std::get<shared_ptr<Material>>(surface)->copyToDevice();
        else
            std::get<shared_ptr<AreaEmitter>>(surface)->copyToDevice();

        // Register data to shader binding table
        HitgroupRecord record;
        prg.recordPackHeader(&record);
        record.data =
        {
            .shape_data = shape->devicePtr(),
            .surface_info =
            {
                .data = is_mat ? std::get<shared_ptr<Material>>(surface)->devicePtr() : std::get<shared_ptr<AreaEmitter>>(surface)->devicePtr(),
                .sample_id = sample_id,
                .type = is_mat ? std::get<shared_ptr<Material>>(surface)->surfaceType() : SurfaceType::AreaEmitter,
            }
        };

        sbt.addHitgroupRecord(record);
        sbt_idx++;
    };

    auto createGAS = [&](shared_ptr<Shape> shape, const Matrix4f& transform, uint32_t num_sbt = 1)
    {
        // Build GAS and add it to IAS
        ShapeInstance instance{ shape->type(), shape, transform };
        instance.allowCompaction();
        instance.buildAccel(context, stream);
        instance.setSBTOffset(sbt_offset);
        instance.setId(instance_id);

        ias.addInstance(instance);

        instance_id++;
        sbt_offset += SBT::NRay * num_sbt;
    };

    auto setupPrimitive = [&](ProgramGroup& prg, const Primitive& p, const Matrix4f& transform)
    {
        addHitgroupRecord(prg, p.shape, p.material, p.sample_id);
        createGAS(p.shape, transform);
    };

    // Textures
    textures.emplace("floor", new CheckerTexture(Vec3f(0.8f), Vec3f(0.3f), 10, checker_prg_id));
    textures.emplace("white", new ConstantTexture(Vec3f(1.0f), constant_prg_id));

    // Materials
    materials.emplace("floor", new Diffuse(textures.at("floor")));

    // Shapes
    shapes.emplace("floor", new Plane(Vec2f(-0.5f), Vec2f(0.5f)));
    shapes.emplace("smoke", new VDBGrid("resources/volume/wdas_cloud_quarter.nvdb", Vec3f(0.8f), Vec3f(0.2f), 0.5f));

    // Floor
    Primitive floor{ shapes.at("floor"), materials.at("floor"), diffuse_prg_id };
    setupPrimitive(plane_prg, floor, Matrix4f::translate(0, -5, 0) * Matrix4f::scale(100));

    // Smoke
    Primitive smoke{ shapes.at("smoke"), materials.at("floor"), medium_prg_id };
    setupPrimitive(grid_prg, smoke, Matrix4f::identity());

    CUDA_CHECK(cudaStreamCreate(&stream));
    ias.build(context, stream);
    params.handle = ias.handle();
    d_params.allocate(sizeof(LaunchParams));
    sbt.createOnDevice();
    pipeline.create(context);
}

// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();

    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

    optixLaunch(
        static_cast<OptixPipeline>(pipeline),
        stream,
        d_params.devicePtr(),
        sizeof(LaunchParams),
        &sbt.sbt(),
        params.width,
        params.height,
        1
    );

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();
    params.frame++;

    result_bmp.copyFromDevice();
}

// ------------------------------------------------------------------
void App::draw()
{
    result_bmp.draw(0, 0);
}

// ------------------------------------------------------------------
void App::mousePressed(float x, float y, int button)
{
    
}

// ------------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    if (button != MouseButton::Middle) return;
    camera_update = true;
}

// ------------------------------------------------------------------
void App::mouseReleased(float x, float y, int button)
{
    
}

// ------------------------------------------------------------------
void App::mouseMoved(float x, float y)
{
    
}

// ------------------------------------------------------------------
void App::mouseScrolled(float x, float y)
{
    camera_update = true;
}

// ------------------------------------------------------------------
void App::keyPressed(int key)
{

}

// ------------------------------------------------------------------
void App::keyReleased(int key)
{

}



