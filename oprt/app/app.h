#pragma once

#include "../core/util.h"
#include "window.h"

namespace oprt {

class BaseApp
{
public:
    BaseApp();
    ~BaseApp();
    
    virtual void setup();
    virtual void update();
    virtual void draw();

    virtual void mousePressed(float x, float y, int button);
    virtual void mouseDragged();
    virtual void mouseMoved();
    virtual void keyPressed();
};

void runApp(std::shared_ptr<Window> window, std::shared_ptr<BaseApp> app);

}