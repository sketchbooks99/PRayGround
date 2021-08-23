#include "app.h"

int main()
{
    auto window = make_shared<Window>("Single GAS", 1920, 1080);
    auto app = make_shared<App>();
    oprtRunApp(app, window);

    return 0;
}