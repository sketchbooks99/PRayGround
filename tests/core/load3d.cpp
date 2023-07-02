#include <prayground/prayground.h>

using namespace std;

int main()
{
    auto filepath = pgFindDataPath("resources/model/Kitchen_set/Kitchen_set.usd");
    if (!filepath)
    {
        pgLogFatal("File is not found:", filepath.value().string());
        return 1;
    }
    Scene<Camera, 1> scene;
    // loadUSDToScene(filepath.value(), scene);

    return 0;
}