#pragma once
#include <prayground/prayground.h>
#include "params.h"
#include <map>

using namespace std;

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
    void mouseScrolled(float xoffset, float yoffset);
private:
    LaunchParams params;
    CUDABuffer<LaunchParams> d_params;
    Pipeline pipeline;
    Context context;
    CUstream stream;
    CornelSBT sbt;
    InstanceAccel ias;

    Bitmap result_bitmap;
    FloatBitmap accum_bitmap;
    Camera camera;

    Instance bunny_instance;
    shared_ptr<EnvironmentEmitter> env;
    vector<pair<shared_ptr<Shape>, shared_ptr<Material>>> cornel_planes;
    shared_ptr<AreaEmitter> area_emitter;
    shared_ptr<Material> bunny_material;
    shared_ptr<TriangleMesh> bunny;
};