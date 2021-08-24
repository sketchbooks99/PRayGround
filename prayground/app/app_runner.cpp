#include "app_runner.h"
#include <chrono>

namespace prayground {

namespace { // nonamed-namespace
    std::shared_ptr<Window> current_window;
    int32_t current_frame{ 0 };
    std::chrono::high_resolution_clock::time_point start_time;
} // ::nonamed-namespace

float pgGetMouseX()
{
    return current_window->events().inputStates.mousePosition.x;
}

float pgGetMouseY()
{
    return current_window->events().inputStates.mousePosition.y;
}

float pgGetPreviousMouseX()
{
    return current_window->events().inputStates.mousePreviousPosition.x;
}

float pgGetPreviousMouseY()
{
    return current_window->events().inputStates.mousePreviousPosition.y;
}

float2 pgGetMousePosition()
{
    return current_window->events().inputStates.mousePosition;
}

float2  pgGetPreviousMousePosition()
{
    return current_window->events().inputStates.mousePreviousPosition;
}

int32_t pgGetMouseButton()
{
    return current_window->events().inputStates.mouseButton;
}

int32_t pgGetWidth()
{
    return current_window->width();
}

int32_t pgGetHeight()
{
    return current_window->height();
}

int32_t pgGetFrame() 
{
    return current_frame;
}

float pgGetFrameRate()
{
    return static_cast<float>(pgGetFrame()) / pgGetElapsedTime<float>();
}

template <typename T>
T pgGetElapsedTime()
{
    std::chrono::duration<T> elapsed = std::chrono::high_resolution_clock::now() - start_time;
    return elapsed.count();
}
template float pgGetElapsedTime<float>();
template double pgGetElapsedTime<double>();

void pgSetWindowName(const std::string& name)
{
    current_window->setName(name);
}

void pgRunApp(const std::shared_ptr<BaseApp>& app, const std::shared_ptr<Window>& window)
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

    start_time = std::chrono::high_resolution_clock::now();
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

        current_frame++;
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

} // ::prayground