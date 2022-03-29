#pragma once

#include <prayground/prayground.h>
#include "params.h"

using namespace std;

using SBT = pgSBT<Camera, 2>;

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
    Pipeline ppl;
    SBT sbt;

    map<string, shared_ptr<Shape>> shapes;
    map<string, shared_ptr<Texture>> textures;
    map<string, shared_ptr<Material>> materials;
    map<string, shared_ptr<AreaEmitter>> lights;
};