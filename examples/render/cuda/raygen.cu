#include "util.cuh"

// Generation of light vertices
extern "C" __device__ void __raygen__lightpath()
{
    VCM vcm = params.vcm;

    const int frame = params.frame;
    const Vec3ui idx(optixGetLaunchIdx());
    uint32_t seed = tea<4>(idx.y() * params.width + idx.x(), frame);

    const int lightID = rndInt(seed, 0, params.num_lights - 1);
    const Vec2f rnd_dir = UniformSampler::get2D(seed);
    const Vec3f rnd_pos = UniformSampler::get3D(seed);

    const AreaEmitterInfo light = params.lights[lightID];

    float radius = vcm.radius;
    radius /= powf(float(vcm.iteration + 1), 0.5f - (1.0f - vcm.radius_alpha));
    radius = fmaxf(radius, 1e-7f);
    const float radius2 = pow2(radius);

    
}

// Connection and merging from camera
extern "C" __device__ void __raygen__camerapath()
{
    const auto* raygen = (pgRaygenData<Camera>*)optixGetSbtDataPointer();

    const int frame = params.frame;
    const Vec3ui idx(optixGetLaunchIndex());
    uint32_t seed = tea<4>(idx.y() * params.width + idx.x(), frame);

    Vec3f result(0.0f);

    for (int i = 0; i < params.samples_per_launch; i++)
    {
        const Vec2f jitter = UniformSampler::get2D(seed) - 0.5f;
        const Vec2f coord(idx.x(), idx.y());
        const Vec2f res(params.width, params.height);

        const Vec2f d = 2.0f * ((coord + jitter) / res) - 1.0f;

        Vec3f ro, rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

        Vec3f throughput(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = Vec3f(0.0f);
        si.albedo = Vec3f(0.0f);
        si.trace_terminate = false;
        si.radiance_evaled = false;

        int depth = 0;
        for ( ;; )
        {
            if (depth >= params.max_depth)
                break;
            
            trace(params.handle, ro, rd, 0.01f, 1e16f, 0, 2, &si);

            if (si.trace_terminate)
            {
                result += si.emission * throughput;
                break;
            }
        }
    }
}