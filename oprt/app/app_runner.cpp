#include "app_runner.h"
#include <chrono>

namespace oprt {

namespace { // nonamed-namespace
    std::shared_ptr<Window> current_window;
    int32_t current_frame = 0;
} // ::nonamed-namespace

float oprtGetMouseX()
{
    return current_window->events().inputStates.mousePosition.x;
}

float oprtGetMouseY()
{
    return current_window->events().inputStates.mousePosition.y;
}

float oprtGetPreviousMouseX()
{
    return current_window->events().inputStates.mousePreviousPosition.x;
}

float oprtGetPreviousMouseY()
{
    return current_window->events().inputStates.mousePreviousPosition.y;
}

float2 oprtGetMousePosition()
{
    return current_window->events().inputStates.mousePosition;
}

float2  oprtGetPreviousMousePosition()
{
    return current_window->events().inputStates.mousePreviousPosition;
}

int32_t oprtGetMouseButton()
{
    return current_window->events().inputStates.mouseButton;
}

int32_t oprtGetWidth()
{
    return current_window->width();
}

int32_t oprtGetHeight()
{
    return current_window->height();
}

void oprtSetWindowName(const std::string& name)
{
    current_window->setName(name);
}

void oprtRunApp(const std::shared_ptr<BaseApp>& app, const std::shared_ptr<Window>& window)
{
    std::shared_ptr<AppRunner> app_runner = std::make_shared<AppRunner>(app, window);
    app_runner->run();
}

// AppRunner ------------------------------------------------
AppRunner::AppRunner(const std::shared_ptr<BaseApp>& app, const std::shared_ptr<Window>& window)
: m_app(app), m_window(window)
{
    current_window = window;
}

// ------------------------------------------------
void AppRunner::run() const
{
    m_window->setup();
    m_app->setup();

    // Register the listener functions
    m_window->events().mousePressed.bindFunction( [&](float x, float y, int button) { return m_app->mousePressed(x, y, button); } );
    m_window->events().mouseDragged.bindFunction( [&](float x, float y, int button) { return m_app->mouseDragged(x, y, button); } );
    m_window->events().mouseReleased.bindFunction( [&](float x, float y, int button) { return m_app->mouseReleased(x, y, button); });
    m_window->events().mouseMoved.bindFunction( [&](float x, float y) { return m_app->mouseMoved(x, y); } );
    m_window->events().mouseScrolled.bindFunction( [&](float xoffset, float yoffset) { return m_app->mouseScrolled(xoffset, yoffset); } );
    m_window->events().keyPressed.bindFunction( [&](int key) { return m_app->keyPressed(key); } );
    m_window->events().keyReleased.bindFunction( [&](int key) { return m_app->keyReleased(key); } );

    m_window->setVisible(true);

    loop();
}

void AppRunner::loop() const 
{
    while (!m_window->shouldClose())
    {
        m_window->update();
        m_app->update();
        m_app->draw();
        m_window->swap();
    }
    close();
}

void AppRunner::close() const
{
    m_app->close();
    m_window->close();
}

// ------------------------------------------------
std::shared_ptr<BaseApp> AppRunner::app() const
{
    return m_app;
}

std::shared_ptr<Window> AppRunner::window() const
{
    return m_window;
}

} // ::oprt