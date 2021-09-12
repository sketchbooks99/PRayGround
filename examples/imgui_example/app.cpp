#include "app.h"

// ----------------------------------------------------------------
void App::setup()
{
    // Initialization of device context.
    stream = 0;
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    context.setDeviceId(0);
    context.create();

    ias = InstanceAccel{ InstanceAccel::Type::Instances };
    ias.allowUpdate();

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

    result_bitmap.allocate(Bitmap::Format::RGBA, pgGetWidth(), pgGetHeight());
    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocate(FloatBitmap::Format::RGBA, pgGetWidth(), pgGetHeight());
    accum_bitmap.allocateDevicePtr();
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.samples_per_launch = 1;
    params.max_depth = 10;
    params.subframe_index = 0;
    params.accum_buffer = reinterpret_cast<float4*>(accum_bitmap.devicePtr());
    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());

    camera.setOrigin(make_float3(0.0f, 0.0f, 40.0f));
    camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFov(40.0f);
    camera.setFovAxis(Camera::FovAxis::Vertical);

    // Create raygen program and bind record;
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, raygen_module, "__raygen__pinhole");
    RaygenRecord raygen_record;
    raygen_prg.recordPackHeader(&raygen_record);
    raygen_record.data.camera.origin = camera.origin();
    raygen_record.data.camera.lookat = camera.lookat();
    raygen_record.data.camera.up = camera.up();
    raygen_record.data.camera.fov = camera.fov();
    raygen_record.data.camera.aspect = 1.0f;
    sbt.setRaygenRecord(raygen_record);

    // SBT record for callable programs
    std::vector<EmptyRecord> callable_records(6, EmptyRecord{});

    // Creating texture programs
    auto [constant_prg, constant_prg_id] = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__constant", "");
    auto [checker_prg, checker_prg_id] = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__checker", "");
    auto [bitmap_prg, bitmap_prg_id] = pipeline.createCallablesProgram(context, textures_module, "__direct_callable__bitmap", "");
    constant_prg.recordPackHeader(&callable_records[0]);
    checker_prg.recordPackHeader(&callable_records[1]);
    bitmap_prg.recordPackHeader(&callable_records[2]);
    sbt.addCallablesRecord(callable_records[0]);
    sbt.addCallablesRecord(callable_records[1]);
    sbt.addCallablesRecord(callable_records[2]);

    // Prepare environment 
    shared_ptr<ConstantTexture> env_texture = make_shared<ConstantTexture>(make_float3(1.0f));
    env_texture->copyToDevice();
    env_texture->setProgramId(constant_prg_id);
    env = make_shared<EnvironmentEmitter>(env_texture);
    env->copyToDevice();

    ProgramGroup miss_prg = pipeline.createMissProgram(context, miss_module, "__miss__envmap");
    MissRecord miss_record;
    miss_prg.recordPackHeader(&miss_record);
    miss_record.data.env_data = env->devicePtr();
    sbt.setMissRecord(miss_record);

    // Preparing textures
    shared_ptr<ConstantTexture> white_constant = make_shared<ConstantTexture>(make_float3(0.8f));
    shared_ptr<ConstantTexture> red_constant = make_shared<ConstantTexture>(make_float3(0.8f, 0.05f, 0.05f));
    shared_ptr<ConstantTexture> green_constant = make_shared<ConstantTexture>(make_float3(0.05f, 0.8f, 0.05f));
    shared_ptr<CheckerTexture> checker = make_shared<CheckerTexture>(make_float3(0.9f), make_float3(0.3f));
    white_constant->setProgramId(constant_prg_id);
    red_constant->setProgramId(constant_prg_id);
    green_constant->setProgramId(constant_prg_id);
    checker->setProgramId(checker_prg_id);
    white_constant->copyToDevice();
    red_constant->copyToDevice();
    green_constant->copyToDevice();
    checker->copyToDevice();

    // Creating material and emitter programs
    auto [diffuse_prg, diffuse_prg_id] = pipeline.createCallablesProgram(context, surfaces_module, "", "__continuation_callable__diffuse");
    auto [dielectric_prg, dielectric_prg_id] = pipeline.createCallablesProgram(context, surfaces_module, "", "__continuation_callable__dielectric");
    auto [area_emitter_prg, area_emitter_prg_id] = pipeline.createCallablesProgram(context, surfaces_module, "__direct_callable__area_emitter", "");
    diffuse_prg.recordPackHeader(&callable_records[3]);
    dielectric_prg.recordPackHeader(&callable_records[4]);
    area_emitter_prg.recordPackHeader(&callable_records[5]);
    sbt.addCallablesRecord(callable_records[3]);
    sbt.addCallablesRecord(callable_records[4]);
    sbt.addCallablesRecord(callable_records[5]);

    // Preparing materials and program id
    shared_ptr<Plane> ceiling_light = make_shared<Plane>(make_float2(-2.5f, -2.5f), make_float2(2.5f, 2.5f));
    shared_ptr<Plane> ceiling = make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f));
    shared_ptr<Plane> right_wall = make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f));
    shared_ptr<Plane> left_wall = make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f));
    shared_ptr<Plane> floor = make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f));
    shared_ptr<Plane> back_wall = make_shared<Plane>(make_float2(-10.0f, -10.0f), make_float2(10.0f, 10.0f));

    shared_ptr<ConstantTexture> emitter_texture = make_shared<ConstantTexture>(make_float3(1.0f));
    emitter_texture->copyToDevice();
    emitter_texture->setProgramId(constant_prg_id);
    area_emitter = make_shared<AreaEmitter>(emitter_texture, 15.0f);
    shared_ptr<Diffuse> ceiling_diffuse = make_shared<Diffuse>(white_constant);
    shared_ptr<Diffuse> right_wall_diffuse = make_shared<Diffuse>(red_constant);
    shared_ptr<Diffuse> left_wall_diffuse = make_shared<Diffuse>(green_constant);
    shared_ptr<Diffuse> floor_diffuse = make_shared<Diffuse>(checker);
    shared_ptr<Diffuse> back_diffuse = make_shared<Diffuse>(white_constant);
    cornel_planes.push_back({ ceiling, ceiling_diffuse });
    cornel_planes.push_back({ right_wall, right_wall_diffuse });
    cornel_planes.push_back({ left_wall, left_wall_diffuse });
    cornel_planes.push_back({ floor, floor_diffuse });
    cornel_planes.push_back({ back_wall, back_diffuse });

    std::vector<Matrix4f> wall_matrices;
    wall_matrices.push_back(Matrix4f::translate({ 0.0f, 10.0f, 0.0f }));                                                              // ceiling
    wall_matrices.push_back(Matrix4f::translate({ 10.0f, 0.0f, 0.0f }) * Matrix4f::rotate(M_PIf / 2.0f, { 0.0f, 0.0f, 1.0f })); // right wall
    wall_matrices.push_back(Matrix4f::translate({ -10.0f, 0.0f, 0.0f }) * Matrix4f::rotate(M_PIf / 2.0f, {0.0f, 0.0f, 1.0f})); // left wall
    wall_matrices.push_back(Matrix4f::translate({ 0.0f, -10.0f, 0.0f }));                                                              // floor
    wall_matrices.push_back(Matrix4f::translate({ 0.0f, 0.0f, -10.0f }) * Matrix4f::rotate(M_PIf / 2.0f, {1.0f, 0.0f, 0.0f})); // back

    ProgramGroup plane_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__plane", "__intersection__plane");

    uint32_t sbt_idx = 0;

    ceiling_light->copyToDevice();
    ceiling_light->setSbtIndex(sbt_idx);
    area_emitter->copyToDevice();
    GeometryAccel gas{ ShapeType::Custom };
    gas.allowCompaction();
    gas.addShape(ceiling_light);
    gas.build(context, stream);
    Instance cl_instance;
    cl_instance.setTransform(Matrix4f::translate({ 0.0f, 9.9f, 0.0f }));
    cl_instance.setTraversableHandle(gas.handle());
    cl_instance.setVisibilityMask(255);
    cl_instance.setSBTOffset(sbt_idx);
    cl_instance.setId(sbt_idx);
    ias.addInstance(cl_instance);

    HitgroupRecord cl_record;
    plane_prg.recordPackHeader(&cl_record);
    cl_record.data.shape_data = ceiling_light->devicePtr();
    cl_record.data.surface_data = area_emitter->devicePtr();
    cl_record.data.surface_type = SurfaceType::AreaEmitter;
    cl_record.data.surface_program_id = area_emitter_prg_id;
    sbt.addHitgroupRecord(cl_record);
    sbt_idx++;

    for (size_t i = 0; i < cornel_planes.size(); i++)
    {
        cornel_planes[i].first->copyToDevice();
        cornel_planes[i].second->copyToDevice();
        cornel_planes[i].first->setSbtIndex(sbt_idx);
        GeometryAccel gas{ShapeType::Custom};
        gas.allowCompaction();
        gas.addShape(cornel_planes[i].first);
        gas.build(context, stream);
        Instance instance;
        instance.setTransform(wall_matrices[i]);
        instance.setTraversableHandle(gas.handle());
        instance.setSBTOffset(sbt_idx);
        instance.setVisibilityMask(255);
        instance.setId(sbt_idx);
        ias.addInstance(instance);

        HitgroupRecord hitgroup_record;
        plane_prg.recordPackHeader(&hitgroup_record);
        hitgroup_record.data.shape_data = cornel_planes[i].first->devicePtr();
        hitgroup_record.data.surface_data = cornel_planes[i].second->devicePtr();
        hitgroup_record.data.surface_type = SurfaceType::Material;
        hitgroup_record.data.surface_program_id = diffuse_prg_id;
        sbt.addHitgroupRecord(hitgroup_record);
        sbt_idx++;
    }

    // Stanford bunny mesh
    bunny = make_shared<TriangleMesh>("uv_bunny.obj");
    bunny->setSbtIndex(sbt_idx);
    bunny->copyToDevice();

    shared_ptr<ConstantTexture> bunny_texture = make_shared<ConstantTexture>(make_float3(1.0f));
    bunny_texture->copyToDevice();
    bunny_texture->setProgramId(constant_prg_id);

    bunny_material = make_shared<Dielectric>(bunny_texture, 1.5f);
    bunny_material->copyToDevice();

    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(context, hitgroups_module, "__closesthit__mesh");

    HitgroupRecord bunny_record; 
    mesh_prg.recordPackHeader(&bunny_record);
    bunny_record.data.shape_data = bunny->devicePtr();
    bunny_record.data.surface_data = bunny_material->devicePtr();
    bunny_record.data.surface_type = SurfaceType::Material;
    bunny_record.data.surface_program_id = dielectric_prg_id;
    sbt.addHitgroupRecord(bunny_record);

    GeometryAccel bunny_gas{ShapeType::Mesh};
    bunny_gas.addShape(bunny);
    bunny_gas.allowCompaction();
    bunny_gas.build(context, stream);
    bunny_instance.setTransform(Matrix4f::translate({0.0f, -5.0f, 0.0f}) * Matrix4f::scale(50.0f));
    bunny_instance.setSBTOffset(sbt_idx);
    bunny_instance.setTraversableHandle(bunny_gas.handle());
    bunny_instance.setVisibilityMask(255);
    bunny_instance.setId(sbt_idx);
    ias.addInstance(bunny_instance);

    // Build IAS
    ias.build(context, stream);
    sbt.createOnDevice();
    params.handle = ias.handle();
    CUDA_CHECK(cudaStreamCreate(&stream));
    pipeline.create(context);
    d_params.allocate(sizeof(LaunchParams));
}

// ----------------------------------------------------------------
void App::update()
{
    params.subframe_index++;
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

    result_bitmap.copyFromDevice();
}

// ----------------------------------------------------------------
void App::draw()
{
    result_bitmap.draw(0, 0);
}

// ----------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    float deltaX = x - pgGetPreviousMousePosition().x;
    float deltaY = y - pgGetPreviousMousePosition().y;
    float cam_length = length(camera.origin());
    float3 cam_dir = normalize(camera.origin() - camera.lookat());

    float theta = acosf(cam_dir.y);
    float phi = atan2(cam_dir.z, cam_dir.x);

    theta = min(math::pi - 0.01f, max(0.01f, theta - math::radians(deltaY * 0.25f)));
    phi += math::radians(deltaX * 0.25f);

    float cam_x = cam_length * sinf(theta) * cosf(phi);
    float cam_y = cam_length * cosf(theta);
    float cam_z = cam_length * sinf(theta) * sinf(phi);

    camera.setOrigin({ cam_x, cam_y, cam_z });

    RaygenRecord* rg_record = reinterpret_cast<RaygenRecord*>(sbt.raygenRecord());
    RaygenData rg_data;
    rg_data.camera =
    {
        .origin = camera.origin(),
        .lookat = camera.lookat(),
        .up = camera.up(),
        .fov = camera.fov(),
        .aspect = 1.0f
    };

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(&rg_record->data),
        &rg_data, sizeof(RaygenData),
        cudaMemcpyHostToDevice
    ));

    CUDA_SYNC_CHECK();

    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocateDevicePtr();
    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<float4*>(accum_bitmap.devicePtr());
    params.subframe_index = 0;

    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);
}

// ----------------------------------------------------------------
void App::mouseScrolled(float xoffset, float yoffset)
{
    float zoom = yoffset < 0 ? 1.1f : 1.0f / 1.1f;
    camera.setOrigin(camera.lookat() + (camera.origin() - camera.lookat()) * zoom);

    RaygenRecord* rg_record = reinterpret_cast<RaygenRecord*>(sbt.raygenRecord());
    RaygenData rg_data;
    rg_data.camera =
    {
        .origin = camera.origin(),
        .lookat = camera.lookat(),
        .up = camera.up(),
        .fov = camera.fov(),
        .aspect = 1.0f
    };

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(&rg_record->data),
        &rg_data, sizeof(RaygenData),
        cudaMemcpyHostToDevice
    ));

    CUDA_SYNC_CHECK();

    result_bitmap.allocateDevicePtr();
    accum_bitmap.allocateDevicePtr();
    params.result_buffer = reinterpret_cast<uchar4*>(result_bitmap.devicePtr());
    params.accum_buffer = reinterpret_cast<float4*>(accum_bitmap.devicePtr());
    params.subframe_index = 0;

    d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);
}