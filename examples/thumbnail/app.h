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
    void close();

    void mouseDragged(float x, float y, int button);
    void mouseScrolled(float xoffset, float yoffset);
private:
    void initResultBufferOnDevice();
    void handleCameraUpdate();

    LaunchParams params;
    CUDABuffer<LaunchParams> d_params;
    Pipeline pipeline;
    Context context;
    CUstream stream;
    SBT sbt;
    InstanceAccel scene_ias;

    Bitmap result_bitmap;
    FloatBitmap accum_bitmap;

    LensCamera camera;
    bool camera_update;

    EnvironmentEmitter env;

    float render_time = 0.0f;

    map<string, shared_ptr<Shape>> shapes;
    map<string, shared_ptr<Material>> materials;
    map<string, shared_ptr<Texture>> textures;
    map<string, shared_ptr<AreaEmitter>> lights;
};