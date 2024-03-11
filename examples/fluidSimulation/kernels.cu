#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

// ------------------------------------------------------------------
// Raygen
// ------------------------------------------------------------------
extern "C" __device__ void __raygen__pinhole() {
    const pgRaygenData<Camera>* raygen = (pgRaygenData<Camera>*)optixGetSbtDataPointer();

    const int frame = params.frame;

    const Vec3ui idx(optixGetLaunchIndex());
    const int image_idx = idx.y() * params.width + idx.x();
    uint32_t seed = tea<4>(image_idx, frame);

    Vec3f result(0.0f);

    int i = params.samples_per_launch;

    while (i > 0) {
        const Vec2f jitter = UniformSampler::get2D(seed) - 0.5f;
        const Vec2f d = 2.0f * Vec2f(
            static_cast<float>(idx.x()) + jitter.x(), 
            static_cast<float>(idx.y()) + jitter.y()
        ) / Vec2f(params.width, params.height) - 1.0f;

        Vec3f ro, rd;

        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

        Vec3f throughput(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = 0.0f;
        si.albedo = 0.0f;
        si.trace_terminate = false;

        int depth = 0;
        for (;;) {
            if (depth >= params.max_depth)
                break;

            uint32_t u0, u1;
            packPointer(si, u0, u1);
            optixTrace(params.handle, ro, rd, 1e-3f, 1e16f, 0.0f, 
                OptixVisibilityMask(255), 
                OPTIX_RAY_FLAG_NONE, 0, 1, 0, 
                u0, u1);

            if (si.trace_terminate) {
                result += throughput * si.emission;
                break;
            }

            // Get emission from area emitter
            if (si.surface_info.type == SurfaceType::AreaEmitter) {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(si.surface_info.callable_id.bsdf, &si, surface_info.data);

                result += throughput * si.emission;
                if (si.trace_terminate)
                    break;
            }
            // Specular sampling
            else if (+(si.surface_info.type & SurfaceType::Delta)) {
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(si.surface_info.callable_id.sammple, &si, surface_info.data);

                // Evaluate BSDF
                Vec3f bsdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(si.surface_info.callable_id.bsdf, &si, surface_info.data);
                throughput *= bsdf;
            }
            // Rough surface sampling
            else {
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(si.surface_info.callable_id.sammple, &si, surface_info.data);

                // Evaluate BSDF
                Vec3f bsdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(si.surface_info.callable_id.bsdf, &si, surface_info.data);

                // Evaluate PDF
                float pdf = optixDirectCall<float, SurfaceInteraction*, void*>(si.surface_info.callable_id.pdf, &si, surface_info.data);
                throughput *= bsdf / pdf;
            }

            ro = si.p;
            rd = si.wi;

            ++depth;
        }
        i--;
    }

    if (!result.isValid()) result = Vec3f(0.0f);

    Vec3f accum_color = result / static_cast<float>(params.samples_per_launch);

    if (frame > 0) {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_color_prev(params.accum_buffer[image_idx]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_idx] = Vec4f(accum_color, 1.0f);
    Vec3u color = make_color(accum_color);
    params.result_buffer[image_idx] = color;
}

// ------------------------------------------------------------------
// Miss 
// ------------------------------------------------------------------
extern "C" __device__ void __miss__envmap() {
    pgMissData* data = (pgMissData*)optixGetSbtDataPointer();
    auto* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    Ray ray = getWorldRay();

    Shading shading;
    float t;
    const Sphere::Data env_sphere{Vec3f(0.0f), 1e16f};
    pgIntersectionSphere(&env_sphere, ray, t, shading);

    si->shading.uv = shading.uv;
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(env->texture.prg_id, si, env->texture.data);
}

// ------------------------------------------------------------------
// Hitgroup program
// ------------------------------------------------------------------
extern "C" __device__ void __intersection__particle() {
    const pgHitgroupData* data = (pgHitgroupData*)optixGetSbtDataPointer();
    const int prim_idx = optixGetPrimitiveIndex();
    const SPHParticle::Data particle = (SPHParticle::Data*)data->shape_data[prim_idx];

    Ray ray = getLocalRay();
    Sphere::Data sphere{particle.position, particle.radius};
    pgIntersectionSphere(&sphere, ray);
}

extern "C" __device__ void __closesthit__custom() {
    const pgHitgroupData* data = (pgHitgroupData*)optixGetSbtDataPointer();

    Ray ray = getWorldRay();

    Shading* shading = getPtrFromTwoAttributes<Shading, 0>();

    // Transform shading frame to world space
    shading->n = normalize(optixTransformNormalFromObjectToWorldSpace(shading->n));
    shading->dpdu = normalize(optixTransformNormalFromObjectToWorldSpace(shading->dpdu));
    shading->dpdv = normalize(optixTransformNormalFromObjectToWorldSpace(shading->dpdv));

    auto* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    si->p = ray.at(ray.tmax);
    si->shading = *shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}
