#include "app.h"

// GUI
#include <prayground/ext/imgui/imgui.h>
#include <prayground/ext/imgui/imgui_impl_glfw.h>
#include <prayground/ext/imgui/imgui_impl_opengl3.h>

#include <random>

void App::initResultBufferOnDevice()
{
    params.frame = 0;

    result_bmp.allocateDevicePtr();
    accum_bmp.allocateDevicePtr();

    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.devicePtr());

    CUDA_SYNC_CHECK();
}

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
    // Initlaize CUDA
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

    const int32_t width = pgGetWidth();
    const int32_t height = pgGetHeight();
    result_bmp.allocate(PixelFormat::RGBA, width, height);
    result_bmp.allocateDevicePtr();
    accum_bmp.allocate(PixelFormat::RGBA, width, height);
    accum_bmp.allocateDevicePtr();

    // Configuration of launch parameters
    params.width = result_bmp.width();
    params.height = result_bmp.height();
    params.samples_per_launch = 1;
    params.max_depth = 3;
    params.frame = 0;
    params.white = 10.0f;
    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.devicePtr());

    // Initialize camera
    shared_ptr<Camera> camera(new Camera);
    camera->setOrigin(10.0f, 5.0f, 0.0f);
    camera->setLookat(0.0f, 5.0f, 0.0f);
    camera->setUp(0, 1, 0);
    camera->setFov(40);
    camera->setAspect((float)width / height);
    camera->enableTracking(pgGetCurrentWindow());
    scene.setCamera(camera);

    // Raygen program
    ProgramGroup rg_prg = pipeline.createRaygenProgram(context, module, "__raygen__restir");
    scene.bindRaygenProgram(rg_prg);

    struct Callables {
        Callables(const pair<ProgramGroup, uint32_t>& callable)
            : program(callable.first), ID(callable.second) {}
        ProgramGroup program;
        uint32_t ID;
    };
    Callables bitmap_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__bitmap", "");
    Callables constant_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__constant", "");
    Callables checker_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__checker", "");
    scene.bindCallablesProgram(bitmap_prg.program);
    scene.bindCallablesProgram(constant_prg.program);
    scene.bindCallablesProgram(checker_prg.program);

    // Create callables program for surfaces
    SurfaceCallableID diffuse_id, disney_id, area_id;
    // Diffuse
    {
        Callables brdf_sample = pipeline.createCallablesProgram(context, module, "__direct_callable__sample_diffuse", "__continuation_callable__brdf_diffuse");
        Callables pdf = pipeline.createCallablesProgram(context, module, "__direct_callable__pdf_diffuse", "");
        diffuse_id = { brdf_sample.ID, brdf_sample.ID, pdf.ID };
        scene.bindCallablesProgram(brdf_sample.program);
        scene.bindCallablesProgram(pdf.program);
    }

    // Disney
    // {
    //     Callables brdf_sample = pipeline.createCallablesProgram(context, module, "__direct_callable__sample_disney", "__continuation_callable__brdf_disney");
    //     Callables pdf = pipeline.createCallablesProgram(context, module, "__direct_callable__pdf_disney", "");
    //     disney_id = { brdf_sample.ID, brdf_sample.ID, pdf.ID };
    //     scene.bindCallablesProgram(brdf_sample.program);
    //     scene.bindCallablesProgram(pdf.program);
    // }

    // Area emitter
    {
        Callables brdf = pipeline.createCallablesProgram(context, module, "__direct_callable__area_emitter", "");
        area_id = { brdf.ID, brdf.ID, brdf.ID };
        scene.bindCallablesProgram(brdf.program);
    }

    // Miss program
    array<ProgramGroup, NRay> miss_prgs;
    miss_prgs[0] = pipeline.createMissProgram(context, module, "__miss__envmap");
    miss_prgs[1] = pipeline.createMissProgram(context, module, "__miss__shadow");
    scene.bindMissPrograms(miss_prgs);

    // Create envmap
    envmap_texture = make_shared<ConstantTexture>(Vec3f(0.0f), constant_prg.ID);
    scene.setEnvmap(envmap_texture);

    // Hitgroup program
    array<ProgramGroup, NRay> mesh_prgs;
    mesh_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh");
    mesh_prgs[1] = pipeline.createHitgroupProgram(context, module, "__closesthit__shadow");

    // Load obj scene
    std::vector<Attributes> material_attribs;
    shared_ptr<TriangleMesh> scene_mesh(new TriangleMesh());
    scene_mesh->loadWithMtl("resources/model/sponza/sponza.obj", material_attribs);

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.mipmapFilterMode = cudaFilterModeLinear;
    tex_desc.normalizedCoords = 1;
    tex_desc.sRGB = 1;

    // Create materials from .obj file
    std::vector<shared_ptr<Material>> scene_materials;
    for (const auto& ma : material_attribs)
    {
        shared_ptr<Texture> texture;
        string diffuse_texname = ma.findOneString("diffuse_texture", "");
        if (!diffuse_texname.empty())
            texture = make_shared<BitmapTexture>(diffuse_texname, tex_desc, bitmap_prg.ID);
        else
            texture = make_shared<ConstantTexture>(ma.findOneVec3f("diffuse", Vec3f(1.0f, 0.0f, 1.0f)), constant_prg.ID);
        auto diffuse = make_shared<Diffuse>(diffuse_id, texture);
        scene_materials.emplace_back(diffuse);
    }

    scene.addObject("Scene mesh", scene_mesh, scene_materials, mesh_prgs, Matrix4f::scale(2));

    // Vertex data to construct triangle mesh of many lights
    vector<LightInfo> light_infos;
    vector<shared_ptr<AreaEmitter>> emitters;
    vector<Vec3f> vertices;
    vector<Vec2f> texcoords;
    vector<Vec3f> normals;
    vector<Face> faces;
    vector<uint32_t> sbt_indices;

    uint32_t seed = tea<4>(0, 0);

    constexpr int NUM_LIGHTS = 100;
    for (int i = 0; i < NUM_LIGHTS; i++)
    {
        LightInfo light;

        Vec3f color(rnd(seed), rnd(seed), rnd(seed));
        float intensity = rnd(seed) * 10.0f;
        light.emission = color * intensity;

        float scale = rnd(seed) * 1.0f;
        Vec3f center = (Vec3f(rnd(seed), rnd(seed) * 0.5f + 0.5f, rnd(seed)) * 2.0f - 1.0f) * 20.0f;
        Vec3f v0 = (Vec3f(rnd(seed), rnd(seed), rnd(seed)) * 2.0f - 1.0f) * scale + center;
        Vec3f v1 = (Vec3f(rnd(seed), rnd(seed), rnd(seed)) * 2.0f - 1.0f) * scale + center;
        Vec3f v2 = (Vec3f(rnd(seed), rnd(seed), rnd(seed)) * 2.0f - 1.0f) * scale + center;
        Vec3f n = normalize(cross(v2 - v0, v1 - v0));
        light.triangle = { v0, v1, v2, n };

        light_infos.push_back(light);

        vertices.push_back(v0); vertices.push_back(v1); vertices.push_back(v2);
        normals.push_back(n); normals.push_back(n); normals.push_back(n);
        texcoords.push_back(Vec2f(0, 0)); texcoords.push_back(Vec2f(0, 1)); texcoords.push_back(Vec2f(1, 1));
        int3 index(i * 3 + 0, i * 3 + 1, i * 3 + 2);
        faces.push_back(Face(index, index, index));
        sbt_indices.push_back((uint32_t)i);

        auto color_texture = make_shared<ConstantTexture>(color, constant_prg.ID);
        emitters.push_back(make_shared<AreaEmitter>(area_id, color_texture, intensity));
    }

    std::shared_ptr<TriangleMesh> light_mesh(new TriangleMesh(vertices, faces, normals, texcoords, sbt_indices));
    scene.addLight("lights", light_mesh, emitters, mesh_prgs, Matrix4f::identity());

    CUDABuffer<LightInfo> d_light_infos;
    d_light_infos.copyToDevice(light_infos);

    CUDA_CHECK(cudaStreamCreate(&stream));
    scene.copyDataToDevice();
    scene.buildAccel(context, stream);
    scene.buildSBT();
    pipeline.create(context);

    params.handle = scene.accelHandle();
    params.lights = d_light_infos.deviceData();
    params.num_lights = static_cast<int>(light_infos.size());

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    const char* glsl_version = "#version 330";
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(pgGetCurrentWindow()->windowPtr(), true);
    ImGui_ImplOpenGL3_Init(glsl_version);
}

// ------------------------------------------------------------------
void App::update()
{
    handleCameraUpdate();

    pgSetWindowName("fps: " + toString(pgGetFrameRate()) + ", frame: " + toString(params.frame));

    scene.launchRay(context, pipeline, params, stream, result_bmp.width(), result_bmp.height(), 1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    params.frame++;

    result_bmp.copyFromDevice();

    if (params.frame == 10000)
    {
        result_bmp.write(pgPathJoin(pgAppDir(), "restir.jpg"));
        pgExit();
    }
}

// ------------------------------------------------------------------
void App::draw()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("ReSTIR GUI");

    auto camera = scene.camera();
    float cam_origin[3] = { camera->origin().x(), camera->origin().y(), camera->origin().z() };
    if (ImGui::SliderFloat3("Camera origin", cam_origin, -1000, 1000))
    {
        camera->setOrigin(cam_origin[0], cam_origin[1], cam_origin[2]);
        is_camera_updated = true;
    }

    float cam_lookat[3] = { camera->lookat().x(), camera->lookat().y(), camera->lookat().z() };
    if (ImGui::SliderFloat3("Camera lookat", cam_lookat, -1000, 1000))
    {
        camera->setLookat(cam_lookat[0], cam_lookat[1], cam_lookat[2]);
        is_camera_updated = true;
    }

    float envmap_color[3] = {envmap_texture->color().x(), envmap_texture->color().y(), envmap_texture->color().z()};
    if (ImGui::ColorEdit3("Envmap color", envmap_color))
    {
        envmap_texture->setColor(Vec3f(envmap_color[0], envmap_color[1], envmap_color[2]));
        scene.updateSBT(+(SBTRecordType::Miss));
        initResultBufferOnDevice();
    }

    if (ImGui::SliderFloat("White", &params.white, 1.0f, 1000.0f))
    {
        initResultBufferOnDevice();
    }

    ImGui::End();

    ImGui::Render();

    result_bmp.draw(0, 0);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
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



