#include "app.h"

int main()
{
    auto window = make_shared<Window>("Dynamic Update Example", 1024, 1024);
    auto app = make_shared<App>();
    pgRunApp(app, window);

    return 0;
}