#include "app.h"

int main(int argc, char* argv[])
{
    pgSetAppDir(APP_DIR);

    auto window = make_shared<Window>("Thumbnail", 1920, 1080);
    auto app = make_shared<App>();

    pgRunApp(app, window);

    return 0;
}