#include <prayground/math/vec.h>
#include <iostream>

using namespace std;
using namespace prayground;

int main()
{
    Vec2f v(10, 20);
    Vec3f v3(make_float2(10, 20), 30);
    Vec3f v3_1(v, 30);
    cout << v.toCUVec() << endl;
    cout << v3.toCUVec() << endl;
    cout << v3_1.toCUVec() << endl;

    cout << normalize(v) << endl;
    cout << cross(Vec3f(1,2,3), Vec3f(4,5,6));
}