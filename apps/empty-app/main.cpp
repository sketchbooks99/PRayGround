#include "app.h"

int main()
{
    auto window = std::make_shared<Window>("Empty example", 1920, 1080);
    auto app = std::make_shared<App>();

    oprtRunApp(app, window);
}