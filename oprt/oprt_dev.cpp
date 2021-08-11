#include <glad/glad.h>

#include <GLFW/glfw3.h>

// Header file describe the scene
#include "scene_config.h"

#include "app/baseapp.h"
#include "app/window.h"

#include "gl/shader.h"

#include "oprt.h"

// ========== Helper functions ==========

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>       File for image output\n";
    std::cerr << "         --launch-samples | -s        Numper of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop              Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>       Set image dimensions; defautlt to 768x768\n";
    std::cerr << "         --help | -h                  Print this usage message\n";
    exit( 0 );
}

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata*/ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}

void streamProgress(int current, int max, float elapsed_time, int progress_length)
{
    std::cout << "\rRendering: [";
    int progress = static_cast<int>(((float)(current+1) / max) * progress_length);
    for (int i = 0; i < progress; i++)
        std::cout << "+";
    for (int i = 0; i < progress_length - progress; i++)
        std::cout << " ";
    std::cout << "]";

    std::cout << " [" << std::fixed << std::setprecision(2) << elapsed_time << "s]";

    float percent = (float)(current) / max;
    std::cout << " (" << std::fixed << std::setprecision(2) << (float)(percent * 100.0f) << "%, ";
    std::cout << "Samples: " << current + 1 << " / " << max << ")" << std::flush;
}

// ========== App ==========
class App : public BaseApp 
{
public:
    App() {}
    void setup() 
    {
        shader.load("shaders/debug.vert", "shaders/debug.frag");

        /// @note debug texture
        vertices = {
            // vertices          // texcoords
            -1.0f, -1.0f, 0.0f,  0.0f, 0.0f, 
            -1.0f,  1.0f, 0.0f,  0.0f, 1.0f,
            1.0f,  -1.0f, 0.0f,  1.0f, 0.0f,
            1.0f,  1.0f, 0.0f,  1.0f, 1.0f
        };
        indices = {
            0, 1, 2, 
            2, 1, 3
        };
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertices.size(), vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), indices.data(), GL_STATIC_DRAW);

        // position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        // texcoords attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }
    void update() 
    {

    }
    void draw() 
    {
        shader.begin();

        glBindVertexArray(VAO); 
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    void mousePressed(float x, float y, int button)
    {
        Message(MSG_NORMAL, "App::mousePressed(): Mouse info:", x, y, button);
    }
    void mouseDragged(float x, float y, int button)
    {
        Message(MSG_NORMAL, "App::mouseDragged(): Mouse info:", x, y, button);
    }
    void mouseReleased(float x, float y, int button)
    {
        Message(MSG_NORMAL, "App::mouseReleased(): Mouse info", x, y, button);
    }
    void mouseMoved(float x, float y)
    {
        Message(MSG_NORMAL, "App::mouseMoved(): Mouse info:", x, y);
    }
    void mouseScrolled(float xoffset, float yoffset)
    {
        Message(MSG_NORMAL, "App::mouseScrolled(): Mouse info", xoffset, yoffset);
    }
    void keyPressed(int key)
    {
        Message(MSG_NORMAL, "App::keyPressed(): Key info:", key);
    }
    void keyReleased(int key)
    {
        Message(MSG_NORMAL, "App::keyReleased(): Key info:", key);
    }
    
private:
    gl::Shader shader;

    std::array<GLfloat, 4*5> vertices;
    std::array<GLuint, 6> indices; 

    GLuint VBO, VAO, EBO;
    GLuint texture;
};

// ========== Main ==========
int main(int argc, char* argv[]) {
    std::shared_ptr<Window> window = std::make_shared<Window>("Path tracer", 1024, 768);
    std::shared_ptr<App> app = std::make_shared<App>();
    oprtRunApp(window, app);

    return 0;
}