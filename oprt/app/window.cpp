#include "window.h"

namespace oprt {

// ----------------------------------------------------------------
Window::Window() 
: Window("", 0, 0)
{

}

Window::Window(const std::string& name) 
: Window(name, 0, 0)
{

}

Window::Window(int32_t width, int32_t height) 
: Window("", width, height)
{

}

Window::Window(const std::string& name, int32_t width, int32_t height)
: m_name(name), m_width(width), m_height(height) 
{

}

// ----------------------------------------------------------------
Window::~Window() 
{

}

// ----------------------------------------------------------------
void Window::setup()
{
    m_events = std::make_unique<WindowEvents>();

    // Initialize GLFW
    if (!glfwInit())
        Throw("oprt::Window::setup(): Failed to initialize GLFW.");
    
    if ((m_gl_version_major == 3 && m_gl_version_minor < 2) || m_gl_version_major < 3)
        Message( MSG_ERROR, "oprt::Window::setup(): The version of OpenGL must supports the programmable renderer (OpenGL 3.2 ~)." );

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, m_gl_version_major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, m_gl_version_minor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window_ptr = glfwCreateWindow(m_width, m_height, m_name.c_str(), nullptr, nullptr);
    // Assert(m_window_ptr, "oprt::Window::setup(): Failed to create GLFW window.");
    if (m_window_ptr == nullptr)
    {
        glfwTerminate();
        Throw("oprt::Window::setup(): Failed to create GLFW window.");
    }

    // Set current window context
    glfwMakeContextCurrent(m_window_ptr);

    // Register the callback functions
    glfwSetMouseButtonCallback(m_window_ptr, _mouseButtonCallback);
    glfwSetCursorPosCallback(m_window_ptr, _cursorPosCallback);
    glfwSetKeyCallback(m_window_ptr, _keyCallback);
    glfwSetScrollCallback(m_window_ptr, _scrollCallback);
    glfwSetFramebufferSizeCallback(m_window_ptr, _resizeCallback);

    // Register oprt::Window pointer
    glfwSetWindowUserPointer(m_window_ptr, this);

    /// No vsync
    /// @note For future work, enable to control frame rate specifying this for the suitable value.
    glfwSwapInterval( 0 ); 

    Message(MSG_NORMAL, "oprt::Window::setup(): window info", m_width, m_height, m_name);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        Throw("oprt::Window::setup(): Failed to initialize GLAD.");
    }
}

// ----------------------------------------------------------------
void Window::update()
{
    glfwPollEvents();
    glfwSwapBuffers(m_window_ptr);
}

// ----------------------------------------------------------------
void Window::close()
{
    glfwTerminate();
}

bool Window::shouldClose() const
{
    return glfwWindowShouldClose(m_window_ptr);
}

// ----------------------------------------------------------------
void Window::setSize(int32_t width, int32_t height)
{
    m_width = width; m_height = height;
}
void Window::setWidth(int32_t width)
{
    m_width = width;
}
void Window::setHeight(int32_t height)
{
    m_height = height;
}
int32_t Window::width() const 
{
    return m_width;
}
int32_t Window::height() const
{
    return m_height;
}

// ----------------------------------------------------------------
void Window::setName(const std::string& name)
{
    m_name = name;
}
std::string Window::name() const
{
    return m_name;
}

// ----------------------------------------------------------------
void Window::setGLVersion(int32_t major, int32_t minor)
{
    m_gl_version_major = major; 
    m_gl_version_minor = minor;
}
int32_t Window::glVersionMajor() const 
{ 
    return m_gl_version_major;
}
int32_t Window::glVersionMinor() const
{
    return m_gl_version_minor;
}
std::tuple<int32_t, int32_t> Window::glVersion() const
{
    return { m_gl_version_major, m_gl_version_minor };
}

// ----------------------------------------------------------------
WindowEvents& Window::events()
{
    return *m_events;
}

// ----------------------------------------------------------------
GLFWwindow* Window::windowPtr()
{
    return m_window_ptr;
}

/*****************************************************************
 Static functions 
*****************************************************************/

// ----------------------------------------------------------------
Window* Window::_getCurrent(GLFWwindow* window)
{
    return static_cast<Window*>(glfwGetWindowUserPointer(window));
}

// ----------------------------------------------------------------
void Window::_mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    Window* current_window = _getCurrent(window);

    double mouse_x, mouse_y;
    glfwGetCursorPos(window, &mouse_x, &mouse_y);

    current_window->events().inputStates.mousePosition = make_float2(mouse_x, mouse_y);
    current_window->events().inputStates.mouseButton = button;

    if (action == GLFW_PRESS)
    {
        current_window->events().mousePressed.invoke(mouse_x, mouse_y, button);
        current_window->events().inputStates.mouseButtonPressed = true;
    }
    else if (action == GLFW_RELEASE)
    {
        current_window->events().mouseReleased.invoke(mouse_x, mouse_y, button);
        current_window->events().inputStates.mouseButtonPressed = false;
    }
}

// ----------------------------------------------------------------
void Window::_cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    Window* current_window = _getCurrent(window);

    if (current_window->events().inputStates.mouseButtonPressed)
        current_window->events().mouseDragged.invoke(xpos, ypos, current_window->events().inputStates.mouseButton);
    else 
        current_window->events().mouseMoved.invoke(xpos, ypos);
}

// ----------------------------------------------------------------
void Window::_keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    Window* current_window = _getCurrent(window);
    
    current_window->events().inputStates.key = key;

    if (action == GLFW_PRESS)
    {
        current_window->events().inputStates.keyButtonPressed = true;
        current_window->events().keyPressed.invoke(key);
    }
    else if (action == GLFW_REPEAT)
    {
        /** Not implemented */
    }
    else if (action == GLFW_RELEASE)
    {
        /** Not implemented */
    }
}

// ----------------------------------------------------------------
void Window::_scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    Window* current_window = _getCurrent(window);

    current_window->events().mouseScrolled.invoke(xoffset, yoffset);
}

// ----------------------------------------------------------------
void Window::_resizeCallback(GLFWwindow* window, int width, int height)
{
    Window* current_window = _getCurrent(window);

    current_window->setWidth(width);
    current_window->setHeight(height);

    glViewport(0, 0, width, height);
}

}