#pragma once
#include <prayground/prayground.h>
#include "params.h"
#include <map>

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
    CornelSBT sbt;
    InstanceAccel ias;

    Film film;
    Camera camera;

    shared_ptr<EnvironmentEmitter> env;
    map<shared_ptr<Shape>, shared_ptr<Material>> primitives;
    shared_ptr<AreaEmitter> ceiling_light;
};