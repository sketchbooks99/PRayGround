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
            packPointer(&si, u0, u1);
            optixTrace(params.handle, ro, rd, 1e-3f, 1e16f, 0.0f, 
                OptixVisibilityMask(1), 
                OPTIX_RAY_FLAG_NONE, 0, 1, 0, 
                u0, u1);

            if (si.trace_terminate) {
                result += throughput * si.emission;
                break;
            }

            // Get emission from area emitter
            if (si.surface_info.type == SurfaceType::AreaEmitter) {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(si.surface_info.callable_id.bsdf, &si, si.surface_info.data);

                result += throughput * si.emission;
                if (si.trace_terminate)
                    break;
            }
            // Specular sampling
            else if (+(si.surface_info.type & SurfaceType::Delta)) {
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(si.surface_info.callable_id.sample, &si, si.surface_info.data);

                // Evaluate BSDF
                Vec3f bsdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(si.surface_info.callable_id.bsdf, &si, si.surface_info.data);
                throughput *= bsdf;
            }
            // Rough surface sampling
            else if (+(si.surface_info.type & SurfaceType::Rough)){
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(si.surface_info.callable_id.sample, &si, si.surface_info.data);

                // Evaluate BSDF
                Vec3f bsdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(si.surface_info.callable_id.bsdf, &si, si.surface_info.data);

                // Evaluate PDF
                float pdf = optixDirectCall<float, SurfaceInteraction*, void*>(si.surface_info.callable_id.pdf, &si, si.surface_info.data);
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
    params.result_buffer[image_idx] = Vec4u(color, 255);
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
    pgIntersectionSphere(&env_sphere, ray, &shading, &t);

    si->shading.uv = shading.uv;
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Vec3f, const Vec2f&, void*>(env->texture.prg_id, si->shading.uv, env->texture.data);
}

// ------------------------------------------------------------------
// Hitgroup program
// ------------------------------------------------------------------
// Particle
extern "C" __device__ void __intersection__particle() {
    const pgHitgroupData* data = (pgHitgroupData*)optixGetSbtDataPointer();
    const int prim_idx = optixGetPrimitiveIndex();
    const SPHParticle::Data particle = reinterpret_cast<SPHParticle::Data*>(data->shape_data)[prim_idx];

    Ray ray = getLocalRay();
    Sphere::Data sphere{particle.position, particle.radius};

    pgReportIntersectionSphere(&sphere, ray);
}

extern "C" __device__ void __closesthit__custom() {
    const pgHitgroupData* data = (pgHitgroupData*)optixGetSbtDataPointer();

    Ray ray = getWorldRay();

    Shading* shading = getPtrFromTwoAttributes<Shading, 0>();

    // Transform shading frame to world space
    shading->n = normalize(optixTransformNormalFromObjectToWorldSpace(shading->n));
    shading->dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(shading->dpdu));
    shading->dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(shading->dpdv));

    auto* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    si->p = ray.at(ray.tmax);
    si->shading = *shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}

// Mesh
extern "C" __device__ void __closesthit__mesh()
{
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const TriangleMesh::Data* mesh_data = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Shading shading = pgGetMeshShading(mesh_data, optixGetTriangleBarycentrics(), optixGetPrimitiveIndex());

    // Transform shading from object to world space
    shading.n = normalize(optixTransformNormalFromObjectToWorldSpace(shading.n));
    shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdu));
    shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdv));

    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();
    si->p = ray.at(ray.tmax);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}

// ------------------------------------------------------------------
// Surfaces
// ------------------------------------------------------------------
// Diffuse
extern "C" __device__ void __direct_callable__sample_diffuse(SurfaceInteraction* si, void* data)
{
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(data);
    si->wi = pgImportanceSamplingDiffuse(diffuse, si->wo, si->shading, si->seed);
    si->trace_terminate = false;
}

extern "C" __device__ Vec3f __direct_callable__bsdf_diffuse(SurfaceInteraction* si, void* data)
{
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(data);
    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(diffuse->texture.prg_id, si->shading.uv, diffuse->texture.data);
    si->albedo = albedo;
    si->emission = Vec3f(0.0f);
    return albedo * pgGetDiffuseBRDF(si->wi, si->shading.n);
}

extern "C" __device__ float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* data)
{
    return pgGetDiffusePDF(si->wi, si->shading.n);
}

extern "C" __device__ Vec3f __direct_callable__area_emitter(SurfaceInteraction* si, void* data)
{
    const auto* area = reinterpret_cast<AreaEmitter::Data*>(data);
    si->trace_terminate = true;
    float is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    if (area->twosided)
    {
        is_emitted = 1.0f;
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);
    }

    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(area->texture.prg_id, si->shading.uv, area->texture.data);
    si->albedo = base;
    si->emission = base * area->intensity * is_emitted;
}

// Textures
extern "C" __device__ Vec3f __direct_callable__bitmap(const Vec2f& uv, void* tex_data) {
    return pgGetBitmapTextureValue<Vec3f>(uv, tex_data);
}

extern "C" __device__ Vec3f __direct_callable__constant(const Vec2f& uv, void* tex_data) {
    return pgGetConstantTextureValue<Vec3f>(uv, tex_data);
}

extern "C" __device__ Vec3f __direct_callable__checker(const Vec2f& uv, void* tex_data) {
    return pgGetCheckerTextureValue<Vec3f>(uv, tex_data);
}