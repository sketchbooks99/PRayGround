#pragma once
#include <prayground/prayground.h>
#include "params.h"

using namespace std;

using RaygenRecord = Record<RaygenData>;
using HitgroupRecord = Record<HitgroupData>;
using MissRecord = Record<MissData>;
using EmptyRecord = Record<EmptyData>;

using MotionBlurSBT = ShaderBindingTable<RaygenRecord, MissRecord, HitgroupRecord, EmptyRecord, EmptyRecord, 1>;

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
    MotionBlurSBT sbt;

    Bitmap result_bitmap;
    Camera camera;

    shared_ptr<EnvironmentEmitter> env;
    shared_ptr<AreaEmitter> area;
    shared_ptr<TriangleMesh> bunny;
    shared_ptr<CheckerTexture> checker_texture;
    shared_ptr<ConstantTexture> env_texture;
};