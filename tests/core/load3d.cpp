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
    vector<Vec3f> vertices;
    vector<Face> faces;
    vector<Vec3f> normals;
    vector<Vec2f> texcoords;
    loadUSD(filepath.value(), vertices, faces, normals, texcoords);

    return 0;
}