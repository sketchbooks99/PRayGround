#include "app.h"

int main()
{
    auto window = std::make_shared<Window>("Empty example", 1024, 768);
    auto app = std::make_shared<App>();

    pgRunApp(app, window);
}