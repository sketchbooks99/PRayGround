#include "app.h"

int main(int argc, char* argv[])
{
	pgSetAppDir(APP_DIR);

	auto window = make_shared<Window>("pgVolume", 1024, 1024);
	auto app = make_shared<App>();

	pgRunApp(app, window);

	return 0;
}