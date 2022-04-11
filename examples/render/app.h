#pragma once

#include <prayground/prayground.h>
#include "params.h"

// For GUI
#include <prayground/ext/imgui/imgui.h>
#include <prayground/ext/imgui/imgui_impl_glfw.h>
#include <prayground/ext/imgui/imgui_impl_opengl3.h>

using namespace std;

using SBT = pgDefaultSBT<Camera, 2>;

class App : public BaseApp {
private:
    using ShapePtr = shared_ptr<Shape>;
    using MaterialPtr = shared_ptr<Material>;
    using AreaEmitterPtr = shared_ptr<AreaEmitter>;
    using TexturePtr = shared_ptr<Texture>;

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
    void launchGenLightVertices();
    void initVCMIteration();

    LaunchParams params;
    CUDABuffer<LaunchParams> d_params;
    Pipeline pipeline;
    Pipeline genlight_pipeline;
    Context context;
    CUstream stream;
    SBT sbt;
    InstanceAccel ias;

    Bitmap result_bmp;
    FloatBitmap accum_bmp;
    
    Camera camera;
    bool camera_update;

    EnvironmentEmitter env;

    float light_gen_time = 0.0f; // Calculation time for light vertices generation
    float camera_time = 0.0f;    // Calculation time for camera path (with VCM, VC, VM)

    map<string, ShapePtr> shapes;
    map<string, TexturePtr> textures;
    map<string, MaterialPtr> materials;
    map<string, AreaEmitterPtr> lights;

    // Path vertices information on host side
    thrust::host_vector<PathVertex> h_light_vertices;
};