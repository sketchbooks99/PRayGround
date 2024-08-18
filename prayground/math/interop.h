#pragma once

namespace prayground {

    INLINE HOSTDEVICE float trilinerInterop(float c[2][2][2], float u, float v, float w)
    {
        float accum = 0.0f;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    accum += (i * u + (j - i) * (1 - u)) *
                        (j * v + (1 - j) * (1 - v)) *
                        (k * w + (1 - k) * (1 - w) * c[i][j][k]);
                }
            }
        }
        return accum;
    }

    INLINE HOSTDEVICE float perlinInterop(Vec3f c[2][2][2], float u, float v, float w)
    {
        float uu = u * u * (3 - 2 * u);
        float vv = v * v * (3 - 2 * v);
        float ww = w * w * (3 - 2 * w);
        float accum = 0.0f;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    Vec3f weight_v{ u - i, v - j, w - k };
                    accum += (i * uu + (1 - i) * (1 - uu))
                        * (j * vv + (1 - j) * (1 - vv))
                        * (k * ww + (1 - k) * (1 - ww))
                        * dot(c[i][j][k], weight_v);
                }
            }
        }
        return accum;
    }

    template <typename T>
    INLINE HOSTDEVICE T barycentricInterop(T v0, T v1, T v2, Vec2f uv)
    {
        return v0 * (1.0f - uv.x() - uv.y()) + v1 * uv.x() + v2 * uv.y();
    }

} // namespace prayground