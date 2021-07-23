#include "baseapp.h"
#include "event.h"

namespace oprt {

BaseApp::BaseApp() {}
BaseApp::~BaseApp() {}

void BaseApp::setup() {}
void BaseApp::update() {}
void BaseApp::draw() {}

void runApp(std::shared_ptr<Window> window, std::shared_ptr<BaseApp> app)
{
    window->setup();
    app->setup();

    // Register the listener functions
    window->events().mousePressed.bindFunction( [&](float x, float y, int button) { return app->mousePressed(x, y, button); } );
    window->events().mouseDragged.bindFunction( [&](float x, float y, int button) { return app->mouseDragged(x, y, button); } );
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
    }

    window->close();
}

}