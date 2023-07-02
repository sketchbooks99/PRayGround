#include "app.h"

int main()
{
    pgSetAppDir(APP_DIR);

    auto window = std::make_shared<Window>("pgVolume", 1080, 1080);
    auto app = std::make_shared<App>();

    pgRunApp(app, window);
}