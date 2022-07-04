#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

struct Reservoir {
    int y;          // The output sample (the index of light)
    float wsum;     // The sum of weights
    float M;        // The number of samples seen so far
    float W;        // Probalistic weight

    void update(int i, float weight, uint32_t& seed)
    {
        wsum += weight;
        M++;
        if (rnd(seed) < (weight / wsum))
            y = i;
    }
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

// p^(x) = \rho(x) * Le(x) * G(x), where \rho(x) = BRDF
static __forceinline__ __device__ float targetPDF(
    const Vec3f& brdf, SurfaceInteraction* si, const Vec3f& to_light, const LightInfo& light)
{
    const float area = length(cross(light.triangle.v1 - light.triangle.v0, light.triangle.v2 - light.triangle.v0)) / 2.0f;
    const float cos_theta = fmaxf(dot(si->shading.n, normalize(to_light)), 0.0f);
    const float d = length(to_light);
    const float G = (d * d) / (area * cos_theta);
    return length(brdf * light.emittance) * G;
}

static __forceinline__ __device__ Vec3f randomSampleOnTriangle(uint32_t& seed, const Triangle& triangle)
{
    // Uniform sampling of barycentric coordinates on a triangle
    Vec2f uv = UniformSampler::get2D(seed);
    return triangle.v0 * (1.0f - uv.x() - uv.y()) + triangle.v1 * uv.x() + triangle.v2 * uv.y();
}

static __forceinlnie__ __device__ Reservoir reservoirImportanceSampling(
    SurfaceInteraction* si, int M, uint32_t& seed)
{
    Reservoir r{0, 0, 0, 0};
    for (int i = 0; i < min(params.num_lights, M); i++)
    {
        // Sample a light
        int light_idx = rndInt(seed, 0, params.num_lights - 1);
        LightInfo light = params.lights[light_idx];
        const float pdf = 1.0f / params.num_lights;
        
        const Vec3f light_p = randomSampleOnTriangle(light.triangle);

        // Get brdf wrt the sampled light
        Vec3f brdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
            si->surface_info.callable_id.bsdf, si, si.surface_info.data, light_p);
        // Get target pdf 
        const float target_pdf = targetPDF(brdf, si, light_p - si->p, light);
        // update reservoir
        r.update(light_idx, target_pdf / pdf, seed);
    }

    LightInfo light = params.lights[r.y];
    const Vec3f light_p = randomSampleOnTriangle(seed, light.triangle);
    Vec3f brdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
        si->surface_info.callable_id.bsdf, si, si.surface_info.data, light_p);    

    r.W = ( ( 1.0f / targetPDF(brdf, si, light_p - si->p, light) ) * ( 1.0f / r.M ) * r.wsum );
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