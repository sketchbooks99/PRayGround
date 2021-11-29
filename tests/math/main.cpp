#include <prayground/math/matrix.h>
#include <prayground/core/util.h>
#include <iostream>

using namespace std;
using namespace prayground;

int main()
{
    Matrix4f transform = Matrix4f::translate({10.0f, 15.0f, 20.0f}) * Matrix4f::rotate(math::pi / 3.0f, {1.0f, 0.7f, 0.0f}) * Matrix4f::scale(5.0f);
    Matrix4f inv_transform = transform.inverse();
    float3 v1{10.0f, 15.0f, 20.0f};

    auto inv_v1 = inv_transform.vectorMul(v1);
    auto _v1 = transform.vectorMul(v1);

    LOG(transform * inv_transform);
    LOG(inv_transform * transform);

    LOG(length(v1), v1);
    LOG(length(inv_v1), inv_v1, transform.vectorMul(inv_v1));
    LOG(length(_v1), _v1, inv_transform.vectorMul(_v1));
}