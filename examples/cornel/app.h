#pragma once
#include <oprt/oprt.h>
#include "params.h"

using namespace std;

class App : public BaseApp
{
public:
    void setup();
    void update();
    void draw();

    void mouseDragged(float x, float y, int button);
private:
    Film film;
    Camera camera;
    LaunchParams params;
    CUDABuffer<LaunchParams> d_params;
    Pipeline pipeline;
    Context context;
    OptixShaderBindingTable sbt;
    CUstream stream;

    shared_ptr<EnvironmentEmitter> env;
    shared_ptr<TriangleMesh> bunny;
    unordered_map<string, shared_ptr<Plane>> cornel;
    unordered_map<string, Instance> instances;
    unordered_map<string, shared_ptr<Material>> materials;
    shared_ptr<AreaEmitter> ceiling_light;
    unordered_map<string, shared_ptr<Texture>> textures;
    InstanceAccel ias;
};