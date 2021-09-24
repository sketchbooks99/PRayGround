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
    void keyPressed(int key);
private:
    void initResultBufferOnDevice();

    LaunchParams params;
    CUDABuffer<LaunchParams> d_params;
    Pipeline pipeline;
    Context context;
    CUstream stream;
    MotionBlurSBT sbt;

    InstanceAccel instance_accel;

    Bitmap result_bitmap;
    FloatBitmap accum_bitmap;
    Camera camera;

    float3 sphere_pos, sphere_prev_pos;

    Transform matrix_transform;

    bool is_move = true;
};