#include "app.h"
#include <prayground/core/interaction.h>

// ----------------------------------------------------------------
void App::setup()
{
    // Initialization of device context.
    stream = 0;
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    context.setDeviceId(0);
    context.create();

    ias = InstanceAccel(InstanceAccel::Type::Instances);

    // Prepare pipeline
    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(4);
    pipeline.setContinuationCallableDepth(4);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);

    // Create modules
    Module raygen_module, miss_module, hitgroups_module, textures_module, surfaces_module;
    raygen_module = pipeline.createModuleFromCudaFile(context, "raygen.cu");
    miss_module = pipeline.createModuleFromCudaFile(context, "miss.cu");
    hitgroups_module = pipeline.createModuleFromCudaFile(context, "hitgroups.cu");
    textures_module = pipeline.createModuleFromCudaFile(context, "textures.cu");
    surfaces_module = pipeline.createModuleFromCudaFile(context, "surfaces.cu");

    // Prepare film to store rendered results.
    film = Film(1024, 1024);
    film.addBitmap("result", Bitmap::Format::RGBA);
    film.addFloatBitmap("accum", FloatBitmap::Format::RGBA);
    film.bitmapAt("result")->allocateDevicePtr();
    film.floatBitmapAt("accum")->allocateDevicePtr();
    params.width = film.width();
    params.height = film.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;
    params.subframe_index = 0;
    params.accum_buffer = reinterpret_cast<float4*>(film.floatBitmapAt("accum")->devicePtr());
    params.result_buffer = reinterpret_cast<uchar4*>(film.bitmapAt("result")->devicePtr());

    camera.setOrigin(make_float3(0.0f, 0.0f, 50.0f));
    camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFov(40.0f);
    camera.setFovAxis(Camera::FovAxis::Vertical);

    // Create raygen program and bind record;
    pipeline.createRaygenProgram(context, raygen_module, "__raygen__pinhole");
    RaygenRecord raygen_record;
    pipeline.bindRaygenRecord(&raygen_record);
    raygen_record.data.camera.origin = camera.origin();
    raygen_record.data.camera.lookat = camera.lookat();
    raygen_record.data.camera.up = camera.up();
    raygen_record.data.camera.fov = camera.fov();
    raygen_record.data.camera.aspect = 1.0f;
    //sbt.setRaygenRecord(raygen_record);
    CUDABuffer<RaygenRecord> d_raygen_record;
    d_raygen_record.copyToDevice(&raygen_record, sizeof(RaygenRecord));
    sbt.raygenRecord = d_raygen_record.devicePtr();

    // SBT record for callable programs
    std::vector<EmptyRecord> callable_records(6, EmptyRecord{});
    //sbt.addCallablesRecord(callable_record);

    // Creating texture programs
    uint32_t constant_prg_id = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__constant", "");
    uint32_t checker_prg_id = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__checker", "");
    uint32_t bitmap_prg_id = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__bitmap", "");
    pipeline.bindCallablesRecord(&callable_records[0], 0);
    pipeline.bindCallablesRecord(&callable_records[1], 1);
    pipeline.bindCallablesRecord(&callable_records[2], 2);

    // Prepare environment 
    env = make_shared<EnvironmentEmitter>("image/earth.jpg");
    env->texture()->setProgramId(bitmap_prg_id);
    env->copyToDevice();

    pipeline.createMissProgram(context, miss_module, "__miss__envmap");
    MissRecord miss_record;
    pipeline.bindMissRecord(&miss_record, 0);
    miss_record.data.env_data = env->devicePtr();
    //sbt.setMissRecord(miss_record);

    CUDABuffer<MissRecord> d_miss_record;
    d_miss_record.copyToDevice(&miss_record, sizeof(MissRecord));
    sbt.missRecordBase = d_miss_record.devicePtr();
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof(MissRecord));

    // Preparing textures
    textures.emplace("checker_texture", make_shared<CheckerTexture>(make_float3(0.9f), make_float3(0.3f)));
    textures.emplace("white_constant", make_shared<ConstantTexture>(make_float3(0.8f)));
    textures.emplace("red_constant", make_shared<ConstantTexture>(make_float3(0.8f, 0.05f, 0.05f)));
    textures.emplace("green_constant", make_shared<ConstantTexture>(make_float3(0.05f, 0.8f, 0.05f)));
    textures.at("checker_texture")->setProgramId(checker_prg_id);
    textures.at("white_constant")->setProgramId(constant_prg_id);
    textures.at("red_constant")->setProgramId(constant_prg_id);
    textures.at("green_constant")->setProgramId(constant_prg_id);
    for (auto texture : textures)
        texture.second->copyToDevice();

    // Creating material and emitter programs
    uint32_t diffuse_prg_id = pipeline.createCallablesProgram(context, surfaces_module, "", "__continuation_callable__diffuse");
    uint32_t dielectric_prg_id = pipeline.createCallablesProgram(context, surfaces_module, "", "__continuation_callable__dielectric");
    uint32_t area_emitter_prg_id = pipeline.createCallablesProgram(context, surfaces_module, "__direct_callable__area_emitter", "");
    pipeline.bindCallablesRecord(&callable_records[3], 3);
    pipeline.bindCallablesRecord(&callable_records[4], 4);
    pipeline.bindCallablesRecord(&callable_records[5], 5);

    CUDABuffer<EmptyRecord> d_callable_records;
    d_callable_records.copyToDevice(callable_records);
    sbt.callablesRecordBase = d_callable_records.devicePtr();
    sbt.callablesRecordCount = static_cast<uint32_t>(callable_records.size());
    sbt.callablesRecordStrideInBytes = static_cast<uint32_t>(sizeof(EmptyRecord));

    // Preparing materials and program id
    ceiling_light = make_shared<AreaEmitter>(make_float3(1.0f), 15.0f);
    materials.emplace("white_diffuse", make_shared<Diffuse>(textures.at("white_constant")));
    materials.emplace("red_diffuse", make_shared<Diffuse>(textures.at("red_constant")));
    materials.emplace("green_diffuse", make_shared<Diffuse>(textures.at("green_constant")));
    materials.emplace("checker_diffuse", make_shared<Diffuse>(textures.at("checker_texture")));
    materials.emplace("glass", make_shared<Dielectric>(make_float3(1.0f), 1.5f));
    ceiling_light->setProgramId(area_emitter_prg_id);
    materials.at("white_diffuse")->addProgramId(diffuse_prg_id);
    materials.at("red_diffuse")->addProgramId(diffuse_prg_id);
    materials.at("green_diffuse")->addProgramId(diffuse_prg_id);
    materials.at("checker_diffuse")->addProgramId(diffuse_prg_id);
    materials.at("glass")->addProgramId(dielectric_prg_id);
    ceiling_light->copyToDevice();
    for (auto material : materials)
        material.second->copyToDevice();

    // Prepare cornel box and construct its gas and instance
    cornel.emplace("ceiling_light", make_shared<Plane>(make_float2(-2.5f, -2.5f), make_float2(2.5f, 2.5f)));
    cornel.emplace("ceiling", make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f)));
    cornel.emplace("right", make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f)));
    cornel.emplace("left", make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f)));
    cornel.emplace("floor", make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f)));
    cornel.emplace("back", make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f)));
    cornel.at("ceiling_light")->attachSurface(ceiling_light);
    cornel.at("ceiling")->attachSurface(materials.at("white_diffuse"));
    cornel.at("right")->attachSurface(materials.at("white_diffuse"));
    cornel.at("left")->attachSurface(materials.at("green_diffuse"));
    cornel.at("floor")->attachSurface(materials.at("checker_diffuse"));
    cornel.at("back")->attachSurface(materials.at("white_diffuse"));

    unordered_map<string, Matrix4f> plane_transforms;
    plane_transforms.emplace("ceiling_light", Matrix4f::translate(make_float3(0.0f, 9.9f, 0.0f)));
    plane_transforms.emplace("ceiling", Matrix4f::translate(make_float3(0.0f, 10.0f, 0.0f)));
    plane_transforms.emplace("right", Matrix4f::translate(make_float3(10.0f, 0.0f, 0.0f)) * Matrix4f::rotate(constants::pi / 2.0f, make_float3(0.0f, 0.0f, 1.0f)));
    plane_transforms.emplace("left", Matrix4f::translate(make_float3(-10.0f, 0.0f, 0.0f)) * Matrix4f::rotate(constants::pi / 2.0f, make_float3(0.0f, 0.0f, 1.0f)));
    plane_transforms.emplace("floor", Matrix4f::translate(make_float3(0.0f, -10.0f, 0.0f)));
    plane_transforms.emplace("back", Matrix4f::translate(make_float3(0.0f, 0.0f, -10.0f)) * Matrix4f::rotate(constants::pi / 2.0f, make_float3(1.0f, 0.0f, 0.0f)));

    pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__plane", "__intersection__plane");

    std::vector<HitgroupRecord> hitgroup_records;
    uint32_t sbt_idx = 0;
    for (auto& plane : cornel)
    {
        plane.second->copyToDevice();
        plane.second->setSbtIndex(sbt_idx);
        GeometryAccel gas{GeometryAccel::Type::Custom};
        gas.allowCompaction();
        gas.addShape(plane.second);
        gas.build(context);
        Instance instance;
        instance.setTransform(plane_transforms.at(plane.first));
        instance.setTraversableHandle(gas.handle());
        instance.setSBTOffset(sbt_idx);
        instance.setVisibilityMask(255);
        instance.setId(sbt_idx);
        ias.addInstance(instance);

        HitgroupRecord hitgroup_record;
        pipeline.bindHitgroupRecord(&hitgroup_record, 0);
        hitgroup_record.data.shape_data = plane.second->devicePtr();
        hitgroup_record.data.surface_type = plane.second->surfaceType();
        switch (plane.second->surfaceType())
        {
            case SurfaceType::Material:
            {
                auto material = std::get<shared_ptr<Material>>(plane.second->surface());
                hitgroup_record.data.surface_data = material->devicePtr();
                hitgroup_record.data.surface_program_id = material->programIdAt(0);
                break;
            }
            case SurfaceType::AreaEmitter:
            {
                auto area_emitter = std::get<shared_ptr<AreaEmitter>>(plane.second->surface());
                hitgroup_record.data.surface_data = area_emitter->devicePtr();
                hitgroup_record.data.surface_program_id = area_emitter->programId();
                break;
            }
        }
        /*sbt.addHitgroupRecord(hitgroup_record);*/
        hitgroup_records.push_back(hitgroup_record);
        sbt_idx++;
    }

    // Stanford bunny mesh
    bunny = make_shared<TriangleMesh>("model/bunny.obj");
    bunny->setSbtIndex(sbt_idx);
    bunny->attachSurface(materials.at("glass"));
    bunny->copyToDevice();

    pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__mesh");

    HitgroupRecord bunny_record; 
    pipeline.bindHitgroupRecord(&bunny_record, 1);
    bunny_record.data.shape_data = bunny->devicePtr();
    bunny_record.data.surface_data = bunny->surfaceDevicePtr();
    bunny_record.data.surface_program_id = materials.at("glass")->programIdAt(0);
    //sbt.addHitgroupRecord(bunny_record);
    hitgroup_records.push_back(bunny_record);

    CUDABuffer<HitgroupRecord> d_hitgroup_records;
    d_hitgroup_records.copyToDevice(hitgroup_records);
    sbt.hitgroupRecordBase = d_hitgroup_records.devicePtr();
    sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_records.size());
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitgroupRecord));

    GeometryAccel bunny_gas{GeometryAccel::Type::Mesh};
    bunny_gas.addShape(bunny);
    bunny_gas.allowCompaction();
    bunny_gas.build(context);
    Instance bunny_instance;
    bunny_instance.setSBTOffset(sbt_idx);
    bunny_instance.setTraversableHandle(bunny_gas.handle());
    bunny_instance.setVisibilityMask(255);
    bunny_instance.setId(sbt_idx);
    ias.addInstance(bunny_instance);

    // Build IAS
    //sbt.createOnDevice();
    ias.build(context);
    params.handle = ias.handle();
    pipeline.create(context);
    CUDA_CHECK(cudaStreamCreate(&stream));
    d_params.allocate(sizeof(LaunchParams));
    // CUDA_CHECK(cudaSetDevice(context.deviceId()));
}

// ----------------------------------------------------------------
void App::update()
{
    //film.bitmapAt("result")->allocateDevicePtr();
    //film.floatBitmapAt("accum")->allocateDevicePtr();
    //params.result_buffer = reinterpret_cast<uchar4*>(film.bitmapAt("result")->devicePtr());
    //params.accum_buffer = reinterpret_cast<float4*>(film.floatBitmapAt("accum")->devicePtr());

    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

    optixLaunch(
        static_cast<OptixPipeline>(pipeline),
        stream,
        d_params.devicePtr(),
        sizeof(LaunchParams),
        &sbt,
        params.width,
        params.height,
        1
    );

    CUDA_CHECK(cudaSetDevice(context.deviceId()));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    film.bitmapAt("result")->copyFromDevice();
}

// ----------------------------------------------------------------
void App::draw()
{
    Message(MSG_WARNING, "Draw called");
    film.bitmapAt("result")->draw(0, 0, film.width(), film.height());
}

// ----------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    
}
