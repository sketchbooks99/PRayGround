#include "app_runner.h"
#include <chrono>

namespace prayground {

    namespace { // nonamed namespace
        struct RunnerState
        {
            std::unique_ptr<AppRunner> runner;
            int32_t current_frame;
            float start_time;
            float frame_rate = 60.0f;
            bool is_fix_fps = false;
            bool is_app_window_initialized = false;
        };
        RunnerState g_state;
    } // nonamed namespace

    float pgGetMouseX()
    {
        return g_state.runner->window()->events().inputStates.mousePosition.x();
    }

    float pgGetMouseY()
    {
        return g_state.runner->window()->events().inputStates.mousePosition.y();
    }

    float pgGetPreviousMouseX()
    {
        return g_state.runner->window()->events().inputStates.mousePreviousPosition.x();
    }

    float pgGetPreviousMouseY()
    {
        return g_state.runner->window()->events().inputStates.mousePreviousPosition.y();
    }

    Vec2f pgGetMousePosition()
    {
        return g_state.runner->window()->events().inputStates.mousePosition;
    }

    Vec2f  pgGetPreviousMousePosition()
    {
        return g_state.runner->window()->events().inputStates.mousePreviousPosition;
    }

    int32_t pgGetMouseButton()
    {
        return g_state.runner->window()->events().inputStates.mouseButton;
    }

    int32_t pgGetKey()
    {
        return g_state.runner->window()->events().inputStates.key;
    }

    int32_t pgGetWidth()
    {
        return g_state.runner->window()->width();
    }

    int32_t pgGetHeight()
    {
        return g_state.runner->window()->height();
    }

    int32_t pgGetFrame() 
    {
        return g_state.current_frame;
    }

    float pgGetFrameRate()
    {
        return static_cast<float>(pgGetFrame()) / pgGetElapsedTimef();
    }

    void pgSetFrameRate(const float fps)
    {
        g_state.is_fix_fps = true;
        g_state.frame_rate = fps;
    }

    float pgGetElapsedTimef()
    {
        return glfwGetTime() - g_state.start_time;
    }

    void pgSetWindowName(const std::string& name)
    {
        g_state.runner->window()->setName(name);
    }

    std::shared_ptr<Window> pgGetCurrentWindow()
    {
        return g_state.runner->window();
    }

    void pgRunApp(const std::shared_ptr<BaseApp>& app, const std::shared_ptr<Window>& window)
    {
        g_state.runner = std::make_unique<AppRunner>(app, window);
        g_state.runner->run();
    }

    bool pgAppWindowInitialized()
    {
        return g_state.is_app_window_initialized;
    }

    void pgExit()
    {
        g_state.runner->window()->notifyShouldClose();
    }

    // AppRunner ------------------------------------------------
    AppRunner::AppRunner(const std::shared_ptr<BaseApp>& app, const std::shared_ptr<Window>& window)
    : m_app(app), m_window(window)
    {

    }

    // ------------------------------------------------
    void AppRunner::run() const
    {
        m_window->setup();
        g_state.is_app_window_initialized = true;

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

        g_state.start_time = glfwGetTime();
        loop();
    }

    void AppRunner::loop() const 
    {
        float lasttime = glfwGetTime();
        while (!m_window->shouldClose())
        {
            m_window->update();
            /// @todo update()をオフスクリーンにする
            m_app->update();
            m_app->draw();
            m_window->swap();

            g_state.current_frame++;
            while (glfwGetTime() < lasttime + 1.0f / g_state.frame_rate && g_state.is_fix_fps)
            {

            }
            lasttime += 1.0f / g_state.frame_rate;
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

} // namespace prayground