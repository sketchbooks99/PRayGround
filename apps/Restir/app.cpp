#include "app.h"

// GUI
#include <prayground/ext/imgui/imgui.h>
#include <prayground/ext/imgui/imgui_impl_glfw.h>
#include <prayground/ext/imgui/imgui_impl_opengl3.h>

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
    params.max_depth = 10;
    params.frame = 0;
    params.result_buffer = reinterpret_cast<Vec4u*>(result_bmp.devicePtr());
    params.accum_buffer = reinterpret_cast<Vec4f*>(accum_bmp.devicePtr());

    // Initialize camera
    shared_ptr<Camera> camera(new Camera);
    camera->setOrigin(78, 10, 84);
    camera->setLookat(85, 11, 75);
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
    Callables diffuse_brdf_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__brdf_diffuse", "");
    Callables disney_brdf_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__brdf_disney", "");
    Callables area_emitter_prg = pipeline.createCallablesProgram(context, module, "__direct_callable__area_emitter", "");
    scene.bindCallablesProgram(diffuse_brdf_prg.program);
    scene.bindCallablesProgram(disney_brdf_prg.program);
scene.bindCallablesProgram(area_emitter_prg.program);

SurfaceCallableID diffuse_id{ diffuse_brdf_prg.ID, diffuse_brdf_prg.ID, diffuse_brdf_prg.ID };
SurfaceCallableID disney_id{ disney_brdf_prg.ID, disney_brdf_prg.ID, disney_brdf_prg.ID };
SurfaceCallableID area_id{ area_emitter_prg.ID, area_emitter_prg.ID, area_emitter_prg.ID };

// Miss program
array<ProgramGroup, NRay> miss_prgs;
miss_prgs[0] = pipeline.createMissProgram(context, module, "__miss__envmap");
miss_prgs[1] = pipeline.createMissProgram(context, module, "__miss__shadow");
scene.bindMissPrograms(miss_prgs);

// Create envmap
scene.setEnvmap(make_shared<ConstantTexture>(Vec3f(10.0f), constant_prg.ID));

// Hitgroup program
array<ProgramGroup, NRay> mesh_prgs;
mesh_prgs[0] = pipeline.createHitgroupProgram(context, module, "__closesthit__mesh");
mesh_prgs[1] = pipeline.createHitgroupProgram(context, module, "__closesthit__shadow");

// Load obj scene
std::vector<Attributes> material_attribs;
shared_ptr<TriangleMesh> scene_mesh(new TriangleMesh());
scene_mesh->loadWithMtl("C:/Users/lunae/Documents/3DScenes/San-Miguel/san-miguel.obj", material_attribs);

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

scene.addObject("Scene mesh", scene_mesh, scene_materials, mesh_prgs, Matrix4f::scale(10));

CUDA_CHECK(cudaStreamCreate(&stream));
scene.copyDataToDevice();
scene.buildAccel(context, stream);
scene.buildSBT();
pipeline.create(context);

params.handle = scene.accelHandle();

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

    scene.launchRay(context, pipeline, params, stream, result_bmp.width(), result_bmp.height(), 1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_SYNC_CHECK();

    params.frame++;

    result_bmp.copyFromDevice();
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
    if (ImGui::SliderFloat3("Camera origin", cam_origin, -500, 500))
    {
        camera->setOrigin(cam_origin[0], cam_origin[1], cam_origin[2]);
        is_camera_updated = true;
    }

    float cam_lookat[3] = { camera->lookat().x(), camera->lookat().y(), camera->lookat().z() };
    if (ImGui::SliderFloat3("Camera lookat", cam_lookat, -100, 100))
    {
        camera->setLookat(cam_lookat[0], cam_lookat[1], cam_lookat[2]);
        is_camera_updated = true;
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



