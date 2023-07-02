#pragma once
#include <prayground/prayground.h>
#include "params.h"
// ImGui
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>

using namespace std;

using RaygenRecord = Record<RaygenData>;
using HitgroupRecord = Record<HitgroupData>;
using MissRecord = Record<MissData>;
using EmptyRecord = Record<EmptyData>;

using SBT = ShaderBindingTable<RaygenRecord, MissRecord, HitgroupRecord, EmptyRecord, EmptyRecord, 1>;

class App : public BaseApp {
public:
    void setup();
    void update();
    void draw();
    void close();

    void keyPressed(int key);
    void mouseDragged(float x, float y, int button);
    void mouseScrolled(float x, float y);
private:
    void initResultBufferOnDevice();
    void handleCameraUpdate();

    LaunchParams params;
    CUDABuffer<LaunchParams> d_params;
    Pipeline pipeline;
    Context context;
    CUstream stream;
    SBT sbt;
    InstanceAccel ias;

    Denoiser denoiser;
    Denoiser::Data denoise_data;

    FloatBitmap result_bitmap, accum_bitmap, albedo_bitmap, normal_bitmap;

    Camera camera;
    bool camera_update;

    EnvironmentEmitter env;

    float render_time, denoise_time;

    map<string, shared_ptr<Shape>> shapes;
    map<string, shared_ptr<Texture>> textures;
    map<string, shared_ptr<Material>> materials;
    map<string, AreaEmitter> lights;
};