#include "app.h"

int main()
{
    auto window = make_shared<Window>("Path tracing", 1440, 720);
    auto app = make_shared<App>();
    pgRunApp(app, window);

    return 0;
}