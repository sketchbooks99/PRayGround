#include "baseapp.h"
#include "event.h"

namespace oprt {

BaseApp::BaseApp() {}
BaseApp::~BaseApp() {}

void BaseApp::setup() {}
void BaseApp::update() {}
void BaseApp::draw() {}

void BaseApp::mousePressed(float x, float y, int button) {}
void BaseApp::mouseDragged(float x, float y, int button) {}
void BaseApp::mouseReleased(float x, float y, int button) {}
void BaseApp::mouseMoved(float x, float y) {}
void BaseApp::mouseScrolled(float xoffset, float yoffset) {}

void BaseApp::keyPressed(int key) {}
void BaseApp::keyReleased(int key) {}

void runApp(const std::shared_ptr<Window>& window, const std::shared_ptr<BaseApp>& app)
{
    window->setup();
    app->setup();

    // Register the listener functions
    window->events().mousePressed.bindFunction( [&](float x, float y, int button) { return app->mousePressed(x, y, button); } );
    window->events().mouseDragged.bindFunction( [&](float x, float y, int button) { return app->mouseDragged(x, y, button); } );
    window->events().mouseReleased.bindFunction( [&](float x, float y, int button) { return app->mouseReleased(x, y, button); });
    window->events().mouseMoved.bindFunction( [&](float x, float y) { return app->mouseMoved(x, y); } );
    window->events().mouseScrolled.bindFunction( [&](float xoffset, float yoffset) { return app->mouseScrolled(xoffset, yoffset); } );
    window->events().keyPressed.bindFunction( [&](int key) { return app->keyPressed(key); } );
    window->events().keyReleased.bindFunction( [&](int key) { return app->keyReleased(key); } );

    /** @todo Implement main loop system */
    while(!glfwWindowShouldClose(window->windowPtr()))
    {
        window->update();
        app->update();

        window->draw();
        app->draw();

        glfwSwapBuffers(window->windowPtr());
        glfwPollEvents();
    }

    window->close();
}

}