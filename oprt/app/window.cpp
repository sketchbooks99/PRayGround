#include "window.h"

namespace oprt {

// ----------------------------------------------------------------
Window::Window() 
{
    states = { false, false, false, false };
}

// ----------------------------------------------------------------
Window::~Window() {}

// ----------------------------------------------------------------
void Window::setup()
{
    if (!glfwInit())
    {
        const char* description;
        glfwGetError(&description);
        Message(MSG_ERROR, description);
        exit(EXIT_FAILURE);
    }
    
    Assert( (m_gl_version_major == 3 && m_gl_version_minor >= 2) || m_gl_version_major >= 4, 
        "oprt::Window::setup(): The version of OpenGL must supports the programmable renderer (OpenGL 3.2 ~).");

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, m_gl_version_major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, m_gl_version_minor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    m_window_ptr = glfwCreateWindow(m_width, m_height, m_name.c_str(), nullptr, nullptr);
    Assert(m_window_ptr, "oprt::Window::setup(): Failed to create GLFW window.");

    glfwMakeContextCurrent(m_window_ptr);

    // Register the callback functions.
    glfwSetMouseButtonCallback(m_window_ptr, _mouseButtonCallback);
    glfwSetCurposCallback(m_window_ptr, _cursorPosCallback);
    glfwSetKeyCallback(m_window_ptr, _keyCallback);
    glfwSetScrollCallback(m_window_ptr, _scrollCallback);
}

// ----------------------------------------------------------------
void Window::update()
{

}

// ----------------------------------------------------------------
void Window::draw()
{

}

// ----------------------------------------------------------------
void Window::mousePressed();

// ----------------------------------------------------------------
void Window::_mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{

}

// ----------------------------------------------------------------
void Window::_cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{

}

// ----------------------------------------------------------------
void Window::_keyCallback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods)
{

}

// ----------------------------------------------------------------
static void Window::_scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{

}

}