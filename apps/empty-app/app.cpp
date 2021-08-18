#include "app.h"

void App::setup()
{
    bitmap.load("016_env.jpg");
    bitmap_draw_w = oprtGetWidth();
    bitmap_draw_h = oprtGetHeight();
}

void App::update()
{

}

void App::draw()
{
    bitmap.draw(0, 0, bitmap_draw_w, bitmap_draw_h);
}

void App::mouseDragged(float x, float y, int button)
{
    bitmap_draw_w = x;
    bitmap_draw_h = y;
}
