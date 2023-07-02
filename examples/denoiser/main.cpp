#include "app.h"

int main()
{
    pgSetAppDir(APP_DIR);

    auto window = make_shared<Window>("Denoiser", 1440, 720);
    auto app = make_shared<App>();

    pgRunApp(app, window);

    return 0;
}