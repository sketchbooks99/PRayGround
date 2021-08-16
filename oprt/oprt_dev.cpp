#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include "oprt.h"
#include "params.h"

using namespace std;

void streamProgress(int current, int max, float elapsed_time, int progress_length)
{
    cout << "\rRendering: [";
    int progress = static_cast<int>(((float)(current+1) / max) * progress_length);
    for (int i = 0; i < progress; i++)
        cout << "+";
    for (int i = 0; i < progress_length - progress; i++)
        cout << " ";
    cout << "]";

    cout << " [" << fixed << setprecision(2) << elapsed_time << "s]";

    float percent = (float)(current) / max;
    cout << " (" << fixed << setprecision(2) << (float)(percent * 100.0f) << "%, ";
    cout << "Samples: " << current + 1 << " / " << max << ")" << flush;
}

// ========== App ==========
class App : public BaseApp 
{
public:
    void setup() 
    {
        // Initialization of device context.
        CUstream stream = 0;
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixInit());
        Context context;
        context.setDeviceId(0);
        context.create();
        
        // Prepare film to store rendered results.
        film = Film(1024, 1024);
        film.addBitmap("result", Bitmap::Format::RGBA);
        film.bitmapAt("result")->copyToDevice();
        params.width = film.width();
        params.height = film.height();
        params.result_buffer = reinterpret_cast<uchar4*>(film.bitmapAt("result")->devicePtr());

        camera.setOrigin(make_float3(0.0f, 0.0f, 50.0f));
        camera.setLookat(make_float3(0.0f, 0.0f, 0.0f));
        camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
        camera.setFov(40.0f);
        camera.setFovAxis(Camera::FovAxis::Vertical);

        env = make_shared<EnvironmentEmitter>("image/earth.jpg");
        env->preparaData();

        // Preparing textures
        textures.emplace("checker_texture", make_shared<CheckerTexture>(make_float3(0.9f), make_float3(0.3f)));
        textures.emplace("white_constant", make_shared<ConstantTexture>(make_float3(0.8f)));
        textures.emplace("red_constant", make_shared<ConstantTexture>(make_float3(0.8f, 0.05f, 0.05f)));
        textures.emplace("green_constant", make_shared<ConstantTexture>(make_float3(0.05f, 0.8f, 0.05f)));
        for (auto texture : textures) 
            texture.second->prepareData();

        // Preparing materials
        ceiling_light = make_shared<AreaEmitter>(make_float3(1.0f), 15.0f);
        materials.emplace("white_diffuse", make_shared<Diffuse>(textures.at("white_constant")));
        materials.emplace("red_diffuse", make_shared<Diffuse>(textures.at("red_constant")));
        materials.emplace("green_diffuse", make_shared<Diffuse>(textures.at("green_constant")));
        materials.emplace("checker_diffuse", make_shared<Diffuse>(textures.at("checker_texture")));
        materials.emplace("glass", make_shared<Dielectric>(make_float3(1.0f), 1.5f));
        ceiling_light->prepareData();
        for (auto material : materials)
            material.second->prepareData();

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

        // Build GAS for cornel box
        uint32_t sbt_base_index = 0;
        unordered_map<string, GeometryAccel> cornel_gases;
        for (auto& plane : cornel)
        {
            cornel.second->prepareData();
            GeometryAccel gas;
            gas.build(context, plane.second, sbt_base_index);
            cornel_gases.emplace(plane.first, gas);
            Instance instance;
            instance.setSBTOffset(sbt_base_index);
            instance.setTraversableHandle(gas.handle());
            instance.setVisibilityMask(255);
            cornel_instance.emplace(plane.first, instance);
            sbt_base_index++;
        }
        // Configuration of instances for cornel box
        cornel_instances.at("ceiling_light").translate(make_float3(0.0f, 9.9f, 0.0f));
        cornel_instances.at("ceiling").translate(make_float3(0.0f, 10.0f, 0.0f));
        cornel_instances.at("right").translate(make_float3(10.0f, 0.0, 0.0f));
        cornel_instances.at("right").rotateZ(M_PIf / 2.0f);
        cornel_instances.at("left").translate(make_float3(-10.0f, 0.0f, 0.0f));
        cornel_instances.at("left").rotateZ(-M_PIf / 2.0f);
        cornel_instances.at("floor").translate(make_float3(0.0f, -10.0f, 0.0f));
        cornel_instances.at("back").translate(make_float3(0.0f, 0.0f, -10.0f));
        cornel_instances.at("back").rotateX(M_PIf / 2.0f);

        bunny = make_shared<TriangleMesh>("model/bunny.obj");
        bunny->attachSurface(materials.at("glass"));
        bunny->prepareData();
        GeometryAccel bunny_gas;
        bunny_gas.build(context, bunny, sbt_base_index);
        Instance bunny_instance;
        bunny_instance.setSBTOffset(sbt_base_index);
        bunny_instance.setTraversableHandle(bunny_gas.handle());

        std::vector<Instance> instances;
        std::transform(cornel_instances.begin(), cornel_instances.end(), std::back_inserter(instances), 
            [](auto& plane) { return plane.second; } );
        instances.push_back(bunny_instance);

        // Build IAS
        InstanceAccel ias;
        ias.build(context, instances);
        params.handle = ias.handle();
        
        // Prepare pipeline
        Pipeline pipeline("params");
        pipeline.setDirectCallableDepth(4);
        pipeline.setContinuationCallableDepth(4);
        pipeline.setNumPayloads(5);
        pipeline.setNumAttributes(5);

        // Create modules
        Module raygen_module, miss_module, hitgroups_module, textures_module, materials_module;
        raygen_module.create(context, "cuda/raygen.cu", pipeline.compileOptions());
        miss_module.create(context, "cuda/miss.cu", pipeline.compileOptions());
        hitgroups_module.create(context, "cuda/hitgroups.cu", pipeline.compileOptions());
        textures_module.create(context, "cuda/textures.cu", pipeline.compileOptions());
        materials_module.create(context, "cuda/materials.cu", pipeline.compileOptions());

        std::vector<ProgramGroup> program_groups;
        // Create raygen program and bind record;
        raygen.createRaygen(context, module, "__raygen__pinhole");
        program_groups.push_back(raygen);
        RaygenRecord raygen_record;
        raygen.bindRecord(&raygen_record);
        raygen_record.data.camera_data.origin = camera.origin();
        raygen_record.data.camera_data.lookat = camera.lookat();
        raygen_record.data.camera_data.up = camera.up();
        raygen_record.data.camera_data.fov = camera.fov();
        CUDABuffer<RaygenRecord> d_raygen_record;
        d_raygen_record.copyToDevice(&raygen_record, sizeof(RaygenRecord));

        miss.createMiss(context, module, "__miss__env");
        program_groups.push_back(miss);
        MissRecord miss_record;
        miss.bindRecord(&miss_record);
        miss_record.data.env_data = env->devicePtr();
        CUDABuffer<MissRecord> d_miss_record;
        d_miss_record.copyToDevice(&miss_record, sizeof(MissRecord));

        ProgramGroup mesh_program, plane_program;
        mesh_program.createHitgroup(context, module, "__closesthit__mesh");
        plane_program.createHitgroup(context, module, "__closesthit__plane", "__intesection__plane");
    }
    void update() 
    {

    }
    void draw() 
    {
        
    }
private:
    Film film;
    Camera camera;
    LaunchParams params;

    shared_ptr<EnvironmentEmitter> env;
    shared_ptr<TriangleMesh> bunny;
    Instance bunny_instance;
    unordered_map<string, shared_ptr<Plane>> cornel;
    unordered_map<string, Instance> cornel_instances;
    unordered_map<string, shared_ptr<Material>> materials;
    shared_ptr<AreaEmitter> ceiling_light;
    unordered_map<string, shared_ptr<Texture>> textures;
    ProgramGroup raygen, miss;
};

// ========== Main ==========
int main(int argc, char* argv[]) {
    shared_ptr<Window> window = make_shared<Window>("Path tracer", 1920, 1080);
    shared_ptr<App> app = make_shared<App>();
    oprtRunApp(app, window);

    return 0;
}