#include "app.h"

int main()
{
    pgSetAppDir(APP_DIR);
    pgAddSearchDir(APP_BINARY_DIR);
    std::cout << APP_BINARY_DIR << std::endl;
    std::cout << std::filesystem::current_path() << std::endl;

    auto window = std::make_shared<Window>("RTCAMP 10", 1920, 1080);
    auto app = std::make_shared<App>();

    pgRunApp(app, window);
}