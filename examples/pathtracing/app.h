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

using PathTracingSBT = ShaderBindingTable<RaygenRecord, MissRecord, HitgroupRecord, EmptyRecord, EmptyRecord, 2>;

class App : public BaseApp
{
public:
    void setup();
    void update();
    void draw();
    void close();

    void mouseDragged(float x, float y, int button);
    void mouseScrolled(float xoffset, float yoffset);
private:
    void initResultBufferOnDevice();

    LaunchParams params;
    CUDABuffer<LaunchParams> d_params;
    Pipeline pipeline;
    Context context;
    CUstream stream;
    PathTracingSBT sbt;
    InstanceAccel scene_ias;

    Bitmap result_bitmap;
    FloatBitmap accum_bitmap;
    FloatBitmap normal_bitmap;
    Camera camera;

    EnvironmentEmitter env;
};