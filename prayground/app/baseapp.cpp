#include "baseapp.h"

namespace prayground {

    BaseApp::BaseApp() {}
    BaseApp::~BaseApp() {}

    void BaseApp::setup() {}
    void BaseApp::update() {}
    void BaseApp::draw() {}
    void BaseApp::close() {}

    void BaseApp::mousePressed(float x, float y, int button) {}
    void BaseApp::mouseDragged(float x, float y, int button) {}
    void BaseApp::mouseReleased(float x, float y, int button) {}
    void BaseApp::mouseMoved(float x, float y) {}
    void BaseApp::mouseScrolled(float xoffset, float yoffset) {}

    void BaseApp::keyPressed(int key) {}
    void BaseApp::keyReleased(int key) {}

}