#include "app.h"
#include <oprt/core/interaction.h>

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
    film.bitmapAt("result")->copyToDevice();
    params.width = film.width();
    params.height = film.height();
    params.samples_per_launch = 10;
    params.max_depth = 10;
    params.subframe_index = 0;
    CUDABuffer<float4> d_accum_buffer;
    d_accum_buffer.allocate(sizeof(float4) * film.bitmapAt("result")->width() * film.bitmapAt("result")->height());
    params.accum_buffer = d_accum_buffer.deviceData();
    params.result_buffer = reinterpret_cast<uchar4*>(film.bitmapAt("result")->devicePtr());

    camera.setOrigin(make_float3(0.0f, 0.0f, 50.0f));
    camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFov(40.0f);
    camera.setFovAxis(Camera::FovAxis::Vertical);

    // Create raygen program and bind record;
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, raygen_module, "__raygen__pinhole");
    RaygenRecord raygen_record;
    raygen_prg.bindRecord(&raygen_record);
    raygen_record.data.camera.origin = camera.origin();
    raygen_record.data.camera.lookat = camera.lookat();
    raygen_record.data.camera.up = camera.up();
    raygen_record.data.camera.fov = camera.fov();
    sbt.setRaygenRecord(raygen_record);

    // Prepare environment 
    env = make_shared<EnvironmentEmitter>("image/earth.jpg");
    env->copyToDevice();

    ProgramGroup miss_prg = pipeline.createMissProgram(context, miss_module, "__miss__envmap");
    MissRecord miss_record;
    miss_prg.bindRecord(&miss_record);
    miss_record.data.env_data = env->devicePtr();
    sbt.setMissRecord(miss_record);

    // SBT record for callable programs
    EmptyRecord callable_record;
    sbt.addCallablesRecord(callable_record);

    // Creating texture programs
    auto [constant_texture_prg, constant_prg_id] = 
        pipeline.createCallablesProgram(context, textures_module, "__direct_callable__constant", "");
    auto [checker_texture_prg, checker_prg_id] = 
        pipeline.createCallablesProgram(context, textures_module, "__direct_callable__checker", "");
    auto [bitmap_texture_prg, bitmap_prg_id] =
        pipeline.createCallablesProgram(context, textures_module, "__direct_callable__bitmap", "");
    constant_texture_prg.bindRecord(&callable_record);
    checker_texture_prg.bindRecord(&callable_record);
    bitmap_texture_prg.bindRecord(&callable_record);

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
    auto [diffuse_prg, diffuse_prg_id] = pipeline.createCallablesProgram(context, surfaces_module, "", "__continuation_callable__diffuse");
    auto [dielectric_prg, dielectric_prg_id] = pipeline.createCallablesProgram(context, surfaces_module, "", "__continuation_callable__dielectric");
    auto [area_emitter_prg, area_emitter_prg_id] = pipeline.createCallablesProgram(context, surfaces_module, "__direct_callable__area_emitter", "");
    diffuse_prg.bindRecord(&callable_record);
    dielectric_prg.bindRecord(&callable_record);
    area_emitter_prg.bindRecord(&callable_record);

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

    ProgramGroup plane_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesethit__plane", "__intersection__plane");

    uint32_t sbt_idx = 0;
    for (auto& plane : cornel)
    {
        plane.second->copyToDevice();
        plane.second->setSbtIndex(sbt_idx);
        GeometryAccel gas{GeometryAccel::Type::Custom};
        gas.addShape(plane.second);
        gas.build(context);
        Instance instance;
        instance.setTraversableHandle(gas.handle());
        instance.setSBTOffset(sbt_idx);
        instance.setVisibilityMask(255);
        ias.addInstance(instance);

        HitgroupRecord hitgroup_record;
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

        plane_prg.bindRecord(&hitgroup_record);
        sbt.addHitgroupRecord(hitgroup_record);
        sbt_idx++;
    }

    // Configuration of instances for cornel box
    instances.at("ceiling_light").translate(make_float3(0.0f, 9.9f, 0.0f));
    instances.at("ceiling").translate(make_float3(0.0f, 10.0f, 0.0f));
    instances.at("right").translate(make_float3(10.0f, 0.0, 0.0f));
    instances.at("right").rotateZ(M_PIf / 2.0f);
    instances.at("left").translate(make_float3(-10.0f, 0.0f, 0.0f));
    instances.at("left").rotateZ(-M_PIf / 2.0f);
    instances.at("floor").translate(make_float3(0.0f, -10.0f, 0.0f));
    instances.at("back").translate(make_float3(0.0f, 0.0f, -10.0f));
    instances.at("back").rotateX(M_PIf / 2.0f);

    // Stanford bunny mesh
    bunny = make_shared<TriangleMesh>("model/bunny.obj");
    bunny->setSbtIndex(sbt_idx);
    bunny->attachSurface(materials.at("glass"));
    bunny->copyToDevice();

    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesethit__mesh");

    HitgroupRecord bunny_record;
    bunny_record.data.shape_data = bunny->devicePtr();
    bunny_record.data.surface_data = bunny->surfaceDevicePtr();
    bunny_record.data.surface_program_id = materials.at("glass")->programIdAt(0);
    mesh_prg.bindRecord(&bunny_record);
    sbt.addHitgroupRecord(bunny_record);

    GeometryAccel bunny_gas{GeometryAccel::Type::Mesh};
    bunny_gas.build(context);
    Instance bunny_instance;
    bunny_instance.setSBTOffset(sbt_idx);
    bunny_instance.setTraversableHandle(bunny_gas.handle());
    bunny_instance.setVisibilityMask(255);
    ias.addInstance(bunny_instance);

    // Build IAS
    ias.build(context);
    params.handle = ias.handle();
    pipeline.create(context);
    d_params.allocate(sizeof(LaunchParams));
}

// ----------------------------------------------------------------
void App::update()
{
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

    CUDA_SYNC_CHECK();

    film.bitmapAt("result")->copyFromDevice();
}

// ----------------------------------------------------------------
void App::draw()
{
    film.bitmapAt("result")->draw(0, 0, film.width(), film.height());
}

// ----------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    
}
