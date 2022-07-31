#pragma once

#include <prayground/prayground.h>
#include "params.h"

using namespace std;

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

    Bitmap result_bmp;
    FloatBitmap accum_bmp;

    static constexpr uint32_t NRay = 2;
    Scene<Camera, NRay> scene;

    shared_ptr<ConstantTexture> envmap_texture;

    bool is_camera_updated;
};