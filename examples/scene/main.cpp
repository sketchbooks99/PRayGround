#include "app.h"

int main()
{
    pgSetAppDir(APP_DIR);

    auto window = std::make_shared<Window>("Scene usage example", 1024, 768);
    auto app = std::make_shared<App>();

    pgRunApp(app, window);
}