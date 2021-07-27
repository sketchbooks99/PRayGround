#include <glad/glad.h>

#include <GLFW/glfw3.h>

// Header file describe the scene
#include "scene_config.h"

#include "app/baseapp.h"
#include "app/window.h"

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

    }
    void update() 
    {

    }
    void draw() 
    {
        // Message(MSG_NORMAL, "Application is running.");
    }

    void mousePressed(float x, float y, int button){}
    void mouseDragged(float x, float y, int button){}
    void mouseReleased(float x, float y, int button){}
    void mouseMoved(float x, float y){}
    void mouseScrolled(float xoffset, float yoffset){}
    void keyPressed(int key) {}
    void keyReleased(int key) {}
    
private:
};

// ========== Main ==========
int main(int argc, char* argv[]) {
    std::shared_ptr<Window> window = std::make_shared<Window>("Path tracer", 1024, 768);
    std::shared_ptr<App> app = std::make_shared<App>();
    runApp(window, app);

    return 0;
}