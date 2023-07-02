#include "app.h"

int main()
{
    pgSetAppDir(APP_DIR);

    auto window = std::make_shared<Window>("Thrust test", 512, 512);
    auto app = std::make_shared<App>();

    pgRunApp(app, window);
}