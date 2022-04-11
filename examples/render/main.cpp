#include "app.h"

int main()
{
    pgSetAppDir(APP_DIR);

    auto window = std::make_shared<Window>("Vertex connection and merging", 1024, 1024);
    auto app = std::make_shared<App>();

    pgRunApp(app, window);
}