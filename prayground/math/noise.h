#pragma once

#include <prayground/math/random.h>
#include <prayground/math/vec.h>

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
                accum += (i*u + (j-i)*(1-u)) * 
                         (j*v + (1-j)*(1-v)) * 
                         (k*w + (1-k)*(1-w)*c[i][j][k]);
            }
        }
    }
    return accum;
}

INLINE HOSTDEVICE float perlinInterop(Vec3f c[2][2][2], float u, float v, float w)
{
    float uu = u*u*(3-2*u);
    float vv = v*v*(3-2*v);
    float ww = w*w*(3-2*w);
    float accum = 0.0f;

    for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
            for(int k=0; k<2; k++) {
                Vec3f weight_v{u-i, v-j, w-k};
                accum += (i*uu + (1-i) * (1-uu))
                        * (j*vv + (1-j) * (1-vv))
                        * (k*ww + (1-k) * (1-ww))
                        * dot(c[i][j][k], weight_v);
            }
        }
    }
    return accum;
}

class PerlinNoise {
public:
    HOSTDEVICE PerlinNoise(unsigned int seed);

    HOSTDEVICE void setSeed(unsigned int seed);

    HOSTDEVICE float turb(const Vec3f& p, int depth=7) const;

    HOSTDEVICE float noise(const Vec3f& p) const;
private:
    static const int POINT_COUNT = 256;
    Vec3f* m_rnd_vec;
    int* m_perm_x;
    int* m_perm_y;
    int* m_perm_z;

    unsigned int m_seed;

    int* perlinGeneratePerm()
    {
        int* p = new int[POINT_COUNT];
        for (int i = 0; i < POINT_COUNT; i++)
            p[i] = i;
        
        permute(m_seed, p, POINT_COUNT);

        return p;
    }

    static void permute(unsigned int& seed, int* p, int n)
    {
        for (int i = n-1; i > 0; i--)
        {
            int target = rndInt(seed, 0, i);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }
};

// Definitions
INLINE HOSTDEVICE PerlinNoise::PerlinNoise(unsigned int seed) : m_seed{seed}
{
    m_rnd_vec = new Vec3f[POINT_COUNT];
    for (int i = 0; i < POINT_COUNT; i++) {
        const Vec3f rnd_v = Vec3f(rnd(seed), rnd(seed), rnd(seed)) * 2.0f - 1.0f;
        m_rnd_vec[i] = normalize(rnd_v);
    }

    m_perm_x = perlinGeneratePerm();
    m_perm_y = perlinGeneratePerm();
    m_perm_z = perlinGeneratePerm();
}

void INLINE HOSTDEVICE PerlinNoise::setSeed(unsigned int seed)
{
    m_seed = seed;
}

float INLINE HOSTDEVICE PerlinNoise::turb(const Vec3f& p, int depth) const 
{
    float accum = 0.0f;
    Vec3f tmp_p = p;
    float weight = 1.0f;

    for (int i = 0; i < depth; i++)
    {
        accum += weight * noise(tmp_p);
        weight *= 0.5f;
        tmp_p *= 2.0f;
    }
    return fabs(accum);
}

float INLINE HOSTDEVICE PerlinNoise::noise(const Vec3f& p) const 
{
    float u = p.x() - floor(p.x());
    float v = p.y() - floor(p.y());
    float w = p.z() - floor(p.z());

    int i = static_cast<int>(floor(p.x()));
    int j = static_cast<int>(floor(p.y()));
    int k = static_cast<int>(floor(p.z()));
    Vec3f c[2][2][2];

    for(int di=0; di<2; di++) {
        for(int dj=0; dj<2; dj++) {
            for(int dk=0; dk<2; dk++) {
                c[di][dj][dk] = m_rnd_vec[
                    m_perm_x[(i+di) & 255] ^ 
                    m_perm_y[(j+dj) & 255] ^ 
                    m_perm_z[(k+dk) & 255]
                ];
            }
        }
    }
    return perlinInterop(c, u, v, w);
}

}