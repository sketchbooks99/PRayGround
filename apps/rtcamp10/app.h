#pragma once

#include <prayground/prayground.h>
#include "params.h"

// GUI
#include <prayground/ext/imgui/imgui.h>
#include <prayground/ext/imgui/imgui_impl_glfw.h>
#include <prayground/ext/imgui/imgui_impl_opengl3.h>

#define SUBMISSION 0

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

    Context ctx;
    CUstream stream;
    Pipeline pipeline;

    LaunchParams params;

    FloatBitmap result_bmp;
    FloatBitmap accum_bmp;
    FloatBitmap normal_bmp;
    FloatBitmap albedo_bmp;

    Denoiser denoiser;
    Denoiser::Data denoise_data;

    static constexpr uint32_t NRay = 2;
    Scene<Camera, NRay> scene;

    bool is_camera_updated;
};