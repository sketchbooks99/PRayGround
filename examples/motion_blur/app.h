#pragma once
#include <prayground/prayground.h>
#include "params.h"
// ImGui
#include <prayground/ext/imgui/imgui.h>
#include <prayground/ext/imgui/imgui_impl_glfw.h>
#include <prayground/ext/imgui/imgui_impl_opengl3.h>

using namespace std;

using RaygenRecord = Record<RaygenData>;
using HitgroupRecord = Record<HitgroupData>;
using MissRecord = Record<MissData>;
using EmptyRecord = Record<EmptyData>;

using SBT = ShaderBindingTable<RaygenRecord, MissRecord, HitgroupRecord, EmptyRecord, EmptyRecord, 1>;

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
    SBT sbt;

    InstanceAccel instance_accel;

    Bitmap result_bitmap;
    FloatBitmap accum_bitmap;
    Camera camera;

    Vec3f sphere_pos, sphere_prev_pos;

    Transform matrix_transform;
    ShapeInstance sphere2;

    bool is_move = true;
};