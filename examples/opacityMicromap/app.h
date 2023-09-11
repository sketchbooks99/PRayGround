#pragma once

#include <prayground/prayground.h>
#include "params.h"

using namespace std;

using RaygenRecord = Record<RaygenData>;
using MissRecord = Record<MissData>;
using HitgroupRecord = Record<HitgroupData>;
using EmptyRecord = Record<EmptyData>;

using SBT = ShaderBindingTable<
    RaygenRecord,
    MissRecord,
    HitgroupRecord,
    EmptyRecord,
    EmptyRecord, 
    1
>;

class App : public BaseApp 
{
public:
    void setup();
    void update();
    void draw();

    void mousePressed(float x, float y, int button);
    void mouseDragged(float x, float y, int button);
    void mouseReleased(float x, float y, int button);
    void mouseMoved(float x, float y);
    void mouseScrolled(float xoffset, float yoffset);

    void keyPressed(int key);
    void keyReleased(int key);
private:
    void initResultBufferOnDevice();
    void handleCameraUpdate();

    Context context;
    CUstream stream;
    Pipeline pipeline;

    LaunchParams params;
    CUDABuffer<LaunchParams> d_params;

    SBT sbt;
    InstanceAccel ias;

    Bitmap result_bmp;
    FloatBitmap accum_bmp;

    EnvironmentEmitter env;

    Camera camera;
    bool is_camera_updated;
};