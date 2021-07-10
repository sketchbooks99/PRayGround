#pragma once

#include "../core/util.h"
#include "../ext/glfw/include/GLFW/glfw3.h"

namespace oprt {

/**
 * @brief 
 * Window manager using GLFW.
 */

class Window {
public:
    struct InputStates
    {
        bool mousePressed;
        bool mouseReleased;
        bool keyPressed;
        bool keyReleased;
    };

    Window();
    ~Window();

    void setup();
    void update();
    void draw();

private:
    GLFWwindow* m_window_ptr = nullptr;
    
    void _mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    void _cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    void _keyCallback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods);
    void _scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    int32_t m_width;
    int32_t m_height;
    std::string m_name;

    int32_t m_gl_version_major;
    int32_t m_gl_version_minor;

    InputStates states;
};

}