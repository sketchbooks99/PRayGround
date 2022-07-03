#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

class Reservoir {
public:
    Reservoir()
    {
        m_sample = Vec3f(0.0f);
        m_weight_sum = 0.0f;
        m_num_strategies = 0;
    }

    void update(Vec3f x, float weight, uint32_t& seed)
    {
        m_weight_sum += weight;
        m_num_strategies += 1;
        if (rnd(seed) < (weight / m_weight_sum))
            m_sample = x;
    }
private:
    Vec3f m_sample;
    float m_weight_sum;
    int32_t m_num_strategies;
};

static __forceinline__ __device__ float calcWeight(const Vec3f& sample)
{
    /// @todo provisional value
    return 1.0f;
}

static __forceinline__ __device__ Reservoir reservoirSampling(int32_t num_strategies, Vec3f sample)
{
    Reservoir r;
    for (int i = 0; i < num_strategies; i++)
        r.update(samples[i], calcWeight(sample), params.seed);
    return r;
}

static __forceinline__ __device__ SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd, 
    float tmin, float tmax, uint32_t ray_type, SurfaceInteraction* si)
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle, ro, rd, 
        tmin, tmax, 0, 
        OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 
        ray_type, 2, ray_type, 
        u0, u1);
}

// raygen 
extern "C" __device__ void __raygen__restir()
{
    const pgRaygenData<Camera> raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int frame = params.frame;
    const Vec3ui idx(optixGetLaunchIndex());
    uint32_t seed = tea<4>(idx.x() * params.width + idx.y(), frame);

    Vec3f result(0.0f);

    int i = params.samples_per_launch;

    do {
        const Vec2f jitter = UniformSampler::get2D(seed) - 0.5f;

        const Vec2f d = 2.0f * Vec2f(
            (static_cast<float>(idx.x()) + jitter.x()) / params.width,
            (static_cast<float>(idx.y()) + jitter.y()) / params.height
        ) - 1.0f;

        Vec3f ro,rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);



    }
}