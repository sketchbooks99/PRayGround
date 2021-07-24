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
    virtual void mouseDragged(float x, float y, int button);
    virtual void mouseMoved(float x, float y);
    virtual void mouseScrolled(float xoffset, float yoffset);
    
    virtual void keyPressed(int key);
    virtual void keyReleased(int key);
};

void runApp(const std::shared_ptr<Window>& window, const std::shared_ptr<BaseApp>& app);

}