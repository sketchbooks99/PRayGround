#include "app.h"

int main(int argc, char* argv[])
{
    pgSetAppDir(APP_DIR);
    pgSetAppName(APP_NAME_DEFINE);

    auto window = make_shared<Window>("Thumbnail", 3840, 2160);
    auto app = make_shared<App>();

    pgRunApp(app, window);

    return 0;
}