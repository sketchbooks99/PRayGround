#include "app.h"

// ------------------------------------------------------------------
void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bitmap.allocateDevicePtr();
    params.result_buffer = reinterpret_cast<Vec4u*>(result_bitmap.devicePtr());

    CUDA_SYNC_CHECK();
}

// ------------------------------------------------------------------
void App::handleCameraUpdate()
{
    if (!is_camera_updated)
        return;

    is_camera_updated = false;

    scene.updateSBT(+(SBTRecordType::Raygen));

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

    pipeline.setLaunchVariableName("params");
    pipeline.setDirectCallableDepth(5);
    pipeline.setContinuationCallableDepth(5);
    pipeline.setNumPayloads(5);
    pipeline.setNumAttributes(5);

    Module module;
    module = pipeline.createModuleFromCudaFile(context, "kernels.cu");

    const uint32_t width = pgGetWidth();
    const uint32_t height = pgGetHeight();
    result_bitmap.allocate(PixelFormat::RGBA, width, height);
    result_bitmap.allocateDevicePtr();

    // Configuration of launch parameters
    params.width = result_bitmap.width();
    params.height = result_bitmap.height();
    params.frame = 0;
    params.result_buffer = reinterpret_cast<Vec4u*>(result_bitmap.devicePtr());

    // Camera settings
    std::shared_ptr<Camera> camera(new Camera);
    camera->setOrigin(0, 300, 500);
    camera->setLookat(0, 0, 0);
    camera->setUp(0, 1, 0);
    camera->setFov(40);
    camera->setAspect((float)width / height);
    camera->enableTracking(pgGetCurrentWindow());

    // Raygen program
    ProgramGroup raygen_prg = pipeline.createRaygenProgram(context, module, "__raygen__shade");
    scene.bindRaygenProgram(raygen_prg);
    scene.setCamera(camera);

    // Create callables program for texture
    struct Callable
    {
        Callable(const std::pair<ProgramGroup, uint32_t>& callable)
            : program(callable.first), ID(callable.second) {}
        ProgramGroup program;
        uint32_t ID;
    };
    Callable bitmap_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__bitmap", "");
    Callable constant_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__constant", "");
    Callable checker_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__checker", "");
    scene.bindCallablesProgram(bitmap_prg.program);
    scene.bindCallablesProgram(constant_prg.program);
    scene.bindCallablesProgram(checker_prg.program);
    
    // Create callable program for phong shading
    Callable phong_bsdf_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__phong", "");
    scene.bindCallablesProgram(phong_bsdf_prg.program);

    // Miss program
    std::array<ProgramGroup, NRay> miss_prgs;
    miss_prgs[0] = pipeline.createMissProgram(context, module, "__miss__envmap");
    scene.bindMissPrograms(miss_prgs);
    scene.setEnvmap(make_shared<ConstantTexture>(Vec3f(0.0f), constant_prg.ID));

    // Hitgroup program
    std::array<ProgramGroup, NRay> mesh_prgs;
    mesh_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh");

    std::array<ProgramGroup, NRay> sphere_prgs;
    sphere_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__sphere");

    std::array<ProgramGroup, NRay> box_prgs;
    box_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__box");

    std::array<ProgramGroup, NRay> plane_prgs;
    plane_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__plane");

    std::array<ProgramGroup, NRay> cylinder_prgs;
    cylinder_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__custom", "__intersection__cylinder");

    using PhongMaterial = CustomMaterial<PhongData>;
    SurfaceType phong_type = SurfaceType::RoughReflection;
    SurfaceCallableID phong_id = { 0, phong_bsdf_prg.ID, 0 };

    // Create phong materials
    PhongData green = {
        .emission = Vec3f(0.0f),
        .ambient = Vec3f(0.1f, 0.2f, 0.1f),
        .diffuse = Vec3f(0.05f, 0.8f, 0.05f),
        .specular = Vec3f(0.9f), 
        .shininess = 0.5f
    };

    PhongData red = {
        .emission = Vec3f(0.0f),
        .ambient = Vec3f(0.2f, 0.1f, 0.1f),
        .diffuse = Vec3f(0.8f, 0.05f, 0.05f),
        .specular = Vec3f(0.9f),
        .shininess = 0.5f
    };

    PhongData blue = {
        .emission = Vec3f(0.0f),
        .ambient = Vec3f(0.1f, 0.1f, 0.2f),
        .diffuse = Vec3f(0.05f, 0.05f, 0.8f),
        .specular = Vec3f(0.9f),
        .shininess = 0.5f
    };

    PhongData white = {
        .emission = Vec3f(0.0f),
        .ambient = Vec3f(0.3f),
        .diffuse = Vec3f(0.7f),
        .specular = Vec3f(0.9f),
        .shininess = 0.5f
    };

    PhongData yellow = {
        .emission = Vec3f(0.0f),
        .ambient = Vec3f(0.3f, 0.3f, 0.1f),
        .diffuse = Vec3f(0.7f, 0.7f, 0.3f),
        .specular = Vec3f(0.9f, 0.9f, 0.4f),
        .shininess = 0.7f
    };

    auto green_phong = make_shared<PhongMaterial>(phong_id, phong_type, green);
    auto red_phong = make_shared<PhongMaterial>(phong_id, phong_type, red);
    auto blue_phong = make_shared<PhongMaterial>(phong_id, phong_type, blue);
    auto white_phong = make_shared<PhongMaterial>(phong_id, phong_type, white);
    auto yellow_phong = make_shared<PhongMaterial>(phong_id, phong_type, yellow);

    scene.addObject("mesh_cylinder", make_shared<CylinderMesh>(1, 2, Vec2ui(30, 30)), green_phong, mesh_prgs, Matrix4f::scale(50.0f));
    scene.addObject("mesh_icosphere", make_shared<IcoSphereMesh>(1, 1), red_phong, mesh_prgs, Matrix4f::translate(-100, 0, 0) * Matrix4f::scale(50.0f));
    scene.addObject("mesh_uvsphere", make_shared<UVSphereMesh>(1, Vec2ui(20, 20)), blue_phong, mesh_prgs, Matrix4f::translate(100, 0, 0)* Matrix4f::scale(50.0f));
    scene.addObject("mesh_plane", make_shared<PlaneMesh>(), white_phong, mesh_prgs, Matrix4f::translate(0, -100, 0)* Matrix4f::scale(100.0f));

    CUDA_CHECK(cudaStreamCreate(&stream));
    scene.copyDataToDevice();
    scene.buildAccel(context, stream);
    scene.buildSBT();
    pipeline.create(context);

    params.handle = scene.accelHandle();
}

// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();

    scene.launchRay(context, pipeline, params, stream, result_bitmap.width(), result_bitmap.height(), 1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    params.frame++;

    result_bitmap.copyFromDevice();
}

// ------------------------------------------------------------------
void App::draw()
{
    result_bitmap.draw();
}

// ------------------------------------------------------------------
void App::mousePressed(float x, float y, int button)
{
    
}

// ------------------------------------------------------------------
void App::mouseDragged(float x, float y, int button)
{
    if (button == MouseButton::Middle) is_camera_updated = true;
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
    is_camera_updated = true;
}

// ------------------------------------------------------------------
void App::keyPressed(int key)
{

}

// ------------------------------------------------------------------
void App::keyReleased(int key)
{

}



