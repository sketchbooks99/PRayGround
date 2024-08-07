#include "app.h"

int main()
{
    pgSetAppDir(APP_DIR);

    auto window = make_shared<Window>("Spectrum", 3840, 2160);
    auto app = make_shared<App>();

    pgRunApp(app, window);

    return 0;
}