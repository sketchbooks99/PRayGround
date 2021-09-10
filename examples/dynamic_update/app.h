#pragma once
#include <prayground/prayground.h>
#include "params.h"

using namespace std;

// Shader Binding Table用のヘッダとデータを格納するRecordクラス
using RaygenRecord = Record<RaygenData>;
using HitgroupRecord = Record<HitgroupData>;
using MissRecord = Record<MissData>;
using EmptyRecord = Record<EmptyData>;

// Rayのタイプは2種類に設定する (0 -> 普通のレイ, 1 -> シャドウレイ)
using DynamicUpdateSBT = ShaderBindingTable<RaygenRecord, MissRecord, HitgroupRecord, EmptyRecord, EmptyRecord, 2>;

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
    DynamicUpdateSBT sbt;
    InstanceAccel ias;

    Bitmap result_bitmap;
    Camera camera;
    
    ShapeInstance sphere1;
    ShapeInstance sphere2;

    vector<pair<ShapeInstance, shared_ptr<Material>>> cornel_box;
    shared_ptr<AreaEmitter> ceiling_emitter;
    shared_ptr<ConstantTexture> ceiling_texture;
};