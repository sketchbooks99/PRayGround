#pragma once

#include <prayground/prayground.h>
#include "params.h"

using namespace std;

using SBT = pgDefaultSBT<Camera, 2>;

class App : public BaseApp {
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

    LaunchParams params;
    CUDABuffer<LaunchParams> d_params;
    Pipeline pipeline;
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

    map<string, shared_ptr<Shape>> shapes;
    map<string, shared_ptr<Texture>> textures;
    map<string, shared_ptr<Material>> materials;
    map<string, shared_ptr<AreaEmitter>> lights;
};