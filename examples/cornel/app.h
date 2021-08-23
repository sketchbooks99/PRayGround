#pragma once
#include <oprt/oprt.h>
#include "params.h"

using namespace std;

// The number of ray types of the application.
//constexpr uint32_t NRay = 1;

using RaygenRecord = Record<RaygenData>;
using HitgroupRecord = Record<HitgroupData>;
using MissRecord = Record<MissData>;
using EmptyRecord = Record<EmptyData>;

using CornelSBT = ShaderBindingTable<RaygenRecord, MissRecord, HitgroupRecord, EmptyRecord, EmptyRecord, 1>;

class App : public BaseApp
{
public:
    void setup();
    void update();
    void draw();

    void mouseDragged(float x, float y, int button);
private:
    LaunchParams params;
    CUDABuffer<LaunchParams> d_params;
    Pipeline pipeline;
    Context context;
    CUstream stream;
    //CornelSBT sbt;
    OptixShaderBindingTable sbt;
    InstanceAccel ias;

    Film film;
    Camera camera;

    shared_ptr<EnvironmentEmitter> env;
    shared_ptr<TriangleMesh> bunny;
    unordered_map<string, shared_ptr<Plane>> cornel;
    unordered_map<string, shared_ptr<Material>> materials;
    shared_ptr<AreaEmitter> ceiling_light;
    unordered_map<string, shared_ptr<Texture>> textures;
};