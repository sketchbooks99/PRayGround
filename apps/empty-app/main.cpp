#include "../oprt/oprt.h"
#include "../oprt/app/window.h"
#include "app.h"

int main()
{
    auto window = std::make_shared<Window>("Path tracer", 1024, 1024);
    auto app = std::make_shared<App>();

    runApp(window, app);
}