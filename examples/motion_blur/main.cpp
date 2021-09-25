#include "app.h"

int main()
{
    pgSetAppDir(APP_DIR);
    
    auto window = make_shared<Window>("Motion Blur", 1024, 1024);
    auto app = make_shared<App>();

    pgRunApp(app, window);

    return 0;
}