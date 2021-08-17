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
        stream = 0;
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixInit());
        context.setDeviceId(0);
        context.create();

        // Prepare pipeline
        pipeline.setLaunchVariableName("params");
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
        surfaces_module.create(context, "cuda/surfaces.cu", pipeline.compileOptions());

        vector<ProgramGroup> program_groups;
        
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

        // Create raygen program and bind record;
        raygen.createRaygen(context, raygen_module, "__raygen__pinhole");
        program_groups.push_back(raygen);
        RaygenRecord raygen_record;
        raygen.bindRecord(&raygen_record);
        raygen_record.data.camera.origin = camera.origin();
        raygen_record.data.camera.lookat = camera.lookat();
        raygen_record.data.camera.up = camera.up();
        raygen_record.data.camera.fov = camera.fov();
        CUDABuffer<RaygenRecord> d_raygen_record;
        d_raygen_record.copyToDevice(&raygen_record, sizeof(RaygenRecord));
        sbt.raygenRecord = d_raygen_record.devicePtr();

        // Prepare environment 
        env = make_shared<EnvironmentEmitter>("image/earth.jpg");
        env->prepareData();

        miss.createMiss(context, miss_module, "__miss__env");
        program_groups.push_back(miss);
        MissRecord miss_record;
        miss.bindRecord(&miss_record);
        miss_record.data.env_data = env->devicePtr();
        CUDABuffer<MissRecord> d_miss_record;
        d_miss_record.copyToDevice(&miss_record, sizeof(MissRecord));
        sbt.missRecordBase = d_miss_record.devicePtr();
        sbt.missRecordCount = 1;
        sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof(MissRecord));

        // SBT record for callable programs
        EmptyRecord callable_record;
        CUDABuffer<EmptyRecord> d_callable_record;
        d_callable_record.copyToDevice(&callable_record, sizeof(EmptyRecord));
        sbt.callablesRecordBase = d_callable_records.devicePtr();
        sbt.callablesRecordCount = 1;
        sbt.callablesRecordStrideInBytes = static_cast<uint32_t>(sizeof(EmptyRecord));

        // Creating texture programs
        ProgramGroup constant_texture_prg, checker_texture_prg, bitmap_texture_prg;
        int32_t constant_prg_id = constant_texture_prg.createCallables(context, textures_module, "__direct_callable__constant", "");
        int32_t checker_prg_id = checker_texture_prg.createCallables(context, textures_module, "__direct_callable__checker", "");
        int32_t bitmap_prg_id = bitmap_texture_prg.createCallables(context, textures_module, "__direct_callable__bitmap", "");
        constant_texture_prg.bindRecord(&callable_record);
        checker_texture_prg.bindRecord(&callable_record);
        bitmap_texture_prg.bindRecord(&callable_record);
        program_groups.push_back(constant_texture_prg);
        program_groups.push_back(checker_texture_prg);
        program_groups.push_back(bitmap_texture_prg);

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
            texture.second->prepareData();

        // Creating material and emitter programs
        ProgramGroup diffuse_prg, dielectric_prg, area_emitter_prg;
        int32_t diffuse_prg_id = diffuse_prg.createCallables(context, surfaces_module, "", "__continuation_callable__diffuse");
        int32_t dielectric_prg_id = dielectric_prg.createCallables(context, surfaces_module, "", "__continuation_callable__dielectric");
        int32_t area_emitter_prg_id = area_emitter_prg.createCallables(context, surfaces_module, "__direct_callable_emitter", "");
        diffuse_prg.bindRecord(&callable_record);
        dielectric_prg.bindRecord(&callable_record);
        area_emitter_prg.bindRecord(&callable_record);
        program_groups.push_back(diffuse_prg);
        program_groups.push_back(dielectric_prg);
        program_groups.push_back(area_emitter_prg);

        // Preparing materials and program id
        ceiling_light = make_shared<AreaEmitter>(make_float3(1.0f), 15.0f);
        materials.emplace("white_diffuse", make_shared<Diffuse>(textures.at("white_constant")));
        materials.emplace("red_diffuse", make_shared<Diffuse>(textures.at("red_constant")));
        materials.emplace("green_diffuse", make_shared<Diffuse>(textures.at("green_constant")));
        materials.emplace("checker_diffuse", make_shared<Diffuse>(textures.at("checker_texture")));
        materials.emplace("glass", make_shared<Dielectric>(make_float3(1.0f), 1.5f));
        ceiling_light->addProgramId(area_emitter_prg_id);
        materials.at("white_diffuse")->addProgramId(diffuse_prg_id);
        materials.at("red_diffuse")->addProgramId(diffuse_prg_id);
        materials.at("green_diffuse")->addProgramId(diffuse_prg_id);
        materials.at("checker_diffuse")->addProgramId(diffuse_prg_id);
        materials.at("glass")->addProgramId(dielectric_prg_id);
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

        ProgramGroup mesh_program, plane_program;
        mesh_program.createHitgroup(context, hitgroups_module, "__closesthit__mesh");
        plane_program.createHitgroup(context, hitgroups_module, "__closesthit__plane", "__intesection__plane");

        // Build GAS for cornel box
        uint32_t sbt_base_index = 0;
        unordered_map<string, GeometryAccel> cornel_gases;
        vector<HitgroupRecord> hitgroup_records;
        for (auto& plane : cornel)
        {
            plane.second->prepareData();
            GeometryAccel gas;
            gas.build(context, plane.second, sbt_base_index);
            cornel_gases.emplace(plane.first, gas);
            Instance instance;
            instance.setSBTOffset(sbt_base_index);
            instance.setTraversableHandle(gas.handle());
            instance.setVisibilityMask(255);
            instances.emplace(plane.first, instance);

            plane.second->addProgram(plane_program);
            program_groups.push_back(plane.second->programAt(0));
            HitgroupRecord hitgroup_record;
            plane.second->bindRecord(&hitgroup_record, 0);
            hitgroup_record.data.shape_data = plane.second->devicePtr();
            hitgroup_record.data.surface_data = plane.second->surfaceDevicePtr();
            switch (plane.second->surfaceType())
            {
                case SurfaceType::Material:
                {
                    auto material = std::get<shared_ptr<Material>>(plane.second->surface());
                    hitgroup_record.data.surface_program_id = material->programIdAt(0);
                    break;
                }
                case SurfaceType::AreaEmitter:
                {
                    auto area_emitter = std::get<shared_ptr<AreaEmitter>>(plane.second->surface());
                    hitgroup_record.data.surface_program_id = area_emitter->programId();
                    break;
                }
            }
            
            hitgroup_records.push_back(hitgroup_record);
            sbt_base_index++;
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
        bunny->attachSurface(materials.at("glass"));
        bunny->prepareData();
        bunny->addProgram(mesh_program);
        program_groups.push_back(bunny->programAt(0));

        HitgroupRecord bunny_record;
        bunny->bindRecord(&bunny_record, 0);
        bunny_record.data.shape_data = bunny->devicePtr();
        bunny_record.data.surface_data = bunny->surfaceDevicePtr();
        bunny_record.data.surface_program_id = materials.at("glass")->programIdAt(0);
        hitgroup_records.push_back(bunny_record);

        // Copy Hitgroup record on device.
        CUDABuffer<HitgroupRecord> d_hitgroup_records;
        d_hitgroup_records.copyToDevice(hitgroup_records);
        sbt.hitgroupRecordBase = d_hitgroup_records.devicePtr();
        sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_records.size());
        sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitgroupRecord));

        GeometryAccel bunny_gas;
        bunny_gas.build(context, bunny, sbt_base_index);
        Instance bunny_instance;
        bunny_instance.setSBTOffset(sbt_base_index);
        bunny_instance.setTraversableHandle(bunny_gas.handle());
        instances.emplace("bunny", bunny_instances);

        std::vector<Instance> instance_arr;
        std::transform(instances.begin(), instances.end(), std::back_inserter(instance_arr), 
            [](auto& plane) { return plane.second; } );

        // Build IAS
        ias.build(context, instances);
        params.handle = ias.handle();

        pipeline.create(context, program_groups);

        d_params.allocate(sizeof(LaunchParams));
    }

    void update() 
    {
        d_params.copyToDeviceAsync(&params, sizeof(LaunchParams), stream);

        optixLaunch(
            static_cast<OptixPipeline>(pipeline), 
            stream, 
            d_params.devicePtr(), 
            sizeof( LaunchParams ), 
            &sbt, 
            params.width, 
            params.height, 
            1
        );

        CUDA_SYNC_CHECK();

        film.bitmapAt("result")->copyFromDevice();
    }

    void draw() 
    {
        film.bitmapAt("result")->draw(0, 0, film.width(), film.height());
    }

    void mouseDragged(float x, float y, int button)
    {

    }

private:
    Film film;
    Camera camera;
    LaunchParams params;
    CUDABuffer<LaunchParams> d_params;
    Pipeline pipeline;
    Context context;
    OptixShaderBindingTable sbt;
    CUstream stream;

    shared_ptr<EnvironmentEmitter> env;
    shared_ptr<TriangleMesh> bunny;
    unordered_map<string, shared_ptr<Plane>> cornel;
    unordered_map<string, Instance> instances;
    unordered_map<string, shared_ptr<Material>> materials;
    shared_ptr<AreaEmitter> ceiling_light;
    unordered_map<string, shared_ptr<Texture>> textures;
    ProgramGroup raygen, miss;
    InstanceAccel ias;
};

// ========== Main ==========
int main(int argc, char* argv[]) {
    shared_ptr<Window> window = make_shared<Window>("Path tracer", 1920, 1080);
    shared_ptr<App> app = make_shared<App>();
    oprtRunApp(app, window);

    return 0;
}