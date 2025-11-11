#include "app.h"


int main()
{
    pgSetAppDir(APP_DIR);

    auto window = std::make_shared<Window>("Ray Tracing Camp 11", 2560, 1440);
    auto app = std::make_shared<App>();

#if SUBMISSION
    pgRunApp(app, window, false);
#else
    pgRunApp(app, window, true);
#endif
}