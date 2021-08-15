#include <glad/glad.h>

#include <GLFW/glfw3.h>

// Header file describe the scene
// #include "scene_config.h"

#include "app/app_runner.h"

#include "gl/shader.h"

#include "oprt.h"

// ========== Helper functions ==========

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
    void setup() 
    {
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixInit());

        Context context;
        context.setDeviceId(0);
        context.create(); 

        
    }
    void update() 
    {

    }
    void draw() 
    {
        
    }
private:
  
};

// ========== Main ==========
int main(int argc, char* argv[]) {
    std::shared_ptr<Window> window = std::make_shared<Window>("Path tracer", 1920, 1080);
    std::shared_ptr<App> app = std::make_shared<App>();
    oprtRunApp(app, window);

    return 0;
}