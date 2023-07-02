#include <prayground/math/vec.h>
#include <vector>
#include <iostream>

using namespace std;
using namespace prayground;

int main()
{
    cout << Vec2f() << endl;
    cout << Vec3f() << endl;
    cout << Vec4f() << endl;

    Vec2f v(10, 20);
    Vec3f v3(make_float2(10, 20), 30);
    Vec3f v3_1(v, 30);
    cout << v.toCUVec() << endl;
    cout << v3.toCUVec() << endl;
    cout << v3_1.toCUVec() << endl;

    cout << normalize(v) << endl;
    cout << cross(Vec3f(1,2,3), Vec3f(4,5,6)) << endl;

    static_assert(sizeof(Vec2f) == sizeof(Vec2f::CUVec));
    static_assert(sizeof(Vec2d) == sizeof(Vec2d::CUVec));
    static_assert(sizeof(Vec2c) == sizeof(Vec2c::CUVec));
    static_assert(sizeof(Vec2s) == sizeof(Vec2s::CUVec));
    static_assert(sizeof(Vec2i) == sizeof(Vec2i::CUVec));
    static_assert(sizeof(Vec2ll) == sizeof(Vec2ll::CUVec));
    static_assert(sizeof(Vec2u) == sizeof(Vec2u::CUVec));
    static_assert(sizeof(Vec2us) == sizeof(Vec2us::CUVec));
    static_assert(sizeof(Vec2ui) == sizeof(Vec2ui::CUVec));
    static_assert(sizeof(Vec2ull) == sizeof(Vec2ull::CUVec));

    static_assert(sizeof(Vec3f) == sizeof(Vec3f::CUVec));
    static_assert(sizeof(Vec3d) == sizeof(Vec3d::CUVec));
    static_assert(sizeof(Vec3c) == sizeof(Vec3c::CUVec));
    static_assert(sizeof(Vec3s) == sizeof(Vec3s::CUVec));
    static_assert(sizeof(Vec3i) == sizeof(Vec3i::CUVec));
    static_assert(sizeof(Vec3ll) == sizeof(Vec3ll::CUVec));
    static_assert(sizeof(Vec3u) == sizeof(Vec3u::CUVec));
    static_assert(sizeof(Vec3us) == sizeof(Vec3us::CUVec));
    static_assert(sizeof(Vec3ui) == sizeof(Vec3ui::CUVec));
    static_assert(sizeof(Vec3ull) == sizeof(Vec3ull::CUVec));

    static_assert(sizeof(Vec4f) == sizeof(Vec4f::CUVec));
    static_assert(sizeof(Vec4d) == sizeof(Vec4d::CUVec));
    static_assert(sizeof(Vec4c) == sizeof(Vec4c::CUVec));
    static_assert(sizeof(Vec4s) == sizeof(Vec4s::CUVec));
    static_assert(sizeof(Vec4i) == sizeof(Vec4i::CUVec));
    static_assert(sizeof(Vec4ll) == sizeof(Vec4ll::CUVec));
    static_assert(sizeof(Vec4u) == sizeof(Vec4u::CUVec));
    static_assert(sizeof(Vec4us) == sizeof(Vec4us::CUVec));
    static_assert(sizeof(Vec4ui) == sizeof(Vec4ui::CUVec));
    static_assert(sizeof(Vec4ull) == sizeof(Vec4ull::CUVec));

    cout << boolalpha;
    cout << is_trivially_copyable_v<Vec4f> << endl;
    cout << is_trivial_v<Vec4f> << endl;

    vector<Vec3f> coords;
    for (int i = 0; i < 10; i++) 
        coords.emplace_back(Vec3f{ (float)i * 3.0f, (float)i * 3 + 1.0f, (float)i * 3 + 2.0f });
    float* data = (float*)coords.data();
    for (int i = 0; i < 30; i++) 
        cout << data[i] << ' ';
    cout << endl;

    Vec3f* raw_coords = reinterpret_cast<Vec3f*>(data);
    for (int i = 0; i < 10; i++) 
        cout << raw_coords[i] << ' ';
    cout << endl;
}