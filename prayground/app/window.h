#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <prayground/math/vec.h>
#include <prayground/core/util.h>
#include <prayground/app/event.h>

namespace prayground {

    /**
     * @brief 
     * Window manager using GLFW.
     */

    struct WindowEvents
    {
        struct InputStates 
        {
            Vec2f mousePosition;
            Vec2f mousePreviousPosition;
            int32_t mouseButton;
            bool mouseButtonPressed;
            int32_t key;
            bool keyButtonPressed;
        };

        InputStates inputStates;

        Event<void, float, float, int>  mousePressed;
        Event<void, float, float, int>  mouseDragged;
        Event<void, float, float, int>  mouseReleased;
        Event<void, float, float>       mouseMoved;
        Event<void, float, float>       mouseScrolled;

        Event<void, int>                keyPressed;
        Event<void, int>                keyReleased;
    };

    class Window {
    public:

        /** @brief Construct a window with the name and the size. */
        Window();
        Window(const std::string& name);
        Window(int32_t width, int32_t height);
        Window(const std::string& name, int32_t width, int32_t height);
        ~Window();

        /** @brief Basical functions of the window. */
        void setup();
        void update();
        void swap();
        void close();
        void notifyShouldClose();
        bool shouldClose() const;

        void setVisible(const bool is_visible);

        /** @brief Setting for the window size. */
        void setSize(int32_t width, int32_t height);
        void setWidth(int32_t width);
        void setHeight(int32_t height);
        int32_t width() const;
        int32_t height() const;

        /** @brief Window name. */
        void setName(const std::string& name);
        std::string name() const;

        /** @brief Setting for the OpenGL version. */
        void setGLVersion(int32_t major, int32_t minor);
        int32_t glVersionMajor() const;
        int32_t glVersionMinor() const;
        std::tuple<int32_t, int32_t> glVersion() const;

        /** @brief Get window events */
        WindowEvents& events();

        GLFWwindow* windowPtr();

    private:
        /** @brief Get current window context from GLFWwindow pointer. */
        static Window* _getCurrent(GLFWwindow* window);

        /** 
         * @brief Callback functions to be binded with the GLFW callback. 
         * @todo Implement the system to capture the current window instance. 
         * */
        static void _mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
        static void _cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
        static void _keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        static void _scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
        static void _resizeCallback(GLFWwindow* window, int width, int height);

        GLFWwindow* m_window_ptr = nullptr;

        std::string m_name;
        int32_t m_width;
        int32_t m_height;

        /** @brief OpenGL version of the window. The version 330 is used at default. */
        int32_t m_gl_version_major = 3;
        int32_t m_gl_version_minor = 3;

        std::unique_ptr<WindowEvents> m_events;
    };

} // namespace prayground