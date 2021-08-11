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
    virtual void close();

    virtual void mousePressed(float x, float y, int button);
    virtual void mouseDragged(float x, float y, int button);
    virtual void mouseReleased(float x, float y, int button);
    virtual void mouseMoved(float x, float y);
    virtual void mouseScrolled(float xoffset, float yoffset);
    
    virtual void keyPressed(int key);
    virtual void keyReleased(int key);
};

}