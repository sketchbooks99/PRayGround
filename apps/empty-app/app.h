#pragma once

#include <oprt/oprt.h>

class App : public BaseApp 
{
public:
    void setup();
    void update();
    void draw();

    void mouseDragged(float x, float y, int button);
private:
    Bitmap bitmap;
    int32_t bitmap_draw_w, bitmap_draw_h;
};