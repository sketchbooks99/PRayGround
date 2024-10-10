#pragma once

#include <prayground/prayground.h>
#include "params.h"

// GUI
#include <prayground/ext/imgui/imgui.h>
#include <prayground/ext/imgui/imgui_impl_glfw.h>
#include <prayground/ext/imgui/imgui_impl_opengl3.h>

#define SUBMISSION 1

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
    std::vector<Keypoint<Vec3f>> cam_pos_keypoints;
    std::vector<Keypoint<Vec3f>> cam_look_keypoints;

    std::vector<PointCloud::Data> awa_pcd_points;
    std::shared_ptr<PointCloud> awa_pcd;
    uint32_t seed;

    float debug_frame = 0.0f;

    bool is_camera_updated;
};