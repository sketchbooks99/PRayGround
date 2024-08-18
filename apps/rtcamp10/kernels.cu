#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

struct LightInteraction {
    /* Surface point on the light source */
    Vec3f p;
    /* Surface normal on the light source */
    Vec3f n;
    /* Texture coordinate on light source */
    Vec2f uv;
    /* Area of light source */
    float area;
    /* PDF of light source */
    float pdf;
    /* Emission from light */
    Vec3f emission;
}

static INLINE DEVICE void trace(
    OptixTraversableHandle handle,
    const Vec3f& ro,
    const Vec3f& rd,
    const float tmin,
    const float tmax,
    SurfaceInteraction* si
) 
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(handle, ro, rd, tmin, tmax, 0.0f, 
        OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 
        0, 2, 0, u0, u1);
}

static INLINE DEVICE bool traceShadowRay(
    OptixTraversableHandle handle,
    const Vec3f& ro, 
    const Vec3f& rd,
    const float tmin,
    const float tmax
) 
{
    uint32_t hit = 0u;
    optixTrace(handle, ro, rd, tmin, tmax, 0.0f, 
        OptixVisibilityMask(1), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 
        1, 2, 1, hit);
    return (bool)hit;
}

static INLINE DEVICE float balanceHeuristic(float pdf1, float pdf2) {
    return pdf1 / (pdf1 + pdf2);
}

static INLINE DEVICE float powerHeuristic(float pdf1, float pdf2) {
    const float p1 = pdf1 * pdf1;
    const float p2 = pdf2 * pdf2;
    return p1 / (p1 + p2);
}

// ----------------------------------------------------------------------------
// Ray generation
// ----------------------------------------------------------------------------
extern "C" DEVICE void __raygen__pinhole() {
    const pgRaygenData<Camera>* rg = (const pgRaygenData<Camera>*)optixGetSbtDataPointer();

    const int frame = params.frame;

    const Vec3ui idx(optixGetLaunchIndex());

    const int image_idx = idx.y() * params.width + idx.x();
    uint32_t seed = tea<4>(image_idx, frame);

    Vec3f result(0.0f);

    int i = params.samples_per_launch;

    while (i > 0) {
        const Vec2f jitter = UniformSampler::get2D(seed) - 0.5f;
        const Vec2f d = 2.0f * Vec2f(
            (float)idx.x() + jitter.x(),
            (float)idx.y() + jitter.y()
        ) / Vec2f(params.width, params.height) - 1.0f;

        Vec3f ro, rd;
        getCameraRay(rg->camera, d.x(), d.y(), ro, rd);

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

            trace(params.handle, ro, rd, 1e-3f, tmax, &si);

            if (si.surface_info.type == SurfaceType::AreaEmitter) {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data
                );
            } 
            // Specular surfaces
            else if (+(si.surface_info.type & SurfaceType::Delta)) {
                // Sample scattered ray
                optixDirectCall<void, SurfaceInteraction*, void*, const Vec3f&>(
                    si.surface_info.callable_id.sample, &si, si.surface_info.data);
                
                // Evaluate BSDF
                Vec3f bsdf = optixDirectCall<void, SurfaceInteraction*, void*, const Vec3f&>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data, si.wi);

                throughput *= bsdf;
            }
            // Rough surface sampling with MIS
            else if (+(si.surface_info.type & SurfaceType::Rough)) {
                LightInfo light;
                if (params.num_lights > 0) {
                    const int light_id = rndInt(si.seed, 0, params.num_lights - 1);
                    light = params.lights[light_id];
                }

                float pdf = 0.0f;

                if (params.num_lights > 0) {
                    LightInteraction li;
                    // Sampling light point
                    optixDirectCall<void, SurfaceInteraction*, LightInfo*, LightInteraction*, void*>(
                        si.surface_info.callable_id.sample, &si, &light, &li, si.surface_info.data
                    );
                    Vec3f to_light = li.p - si.p;
                    const float dist = length(to_light);
                    const Vec3f light_dir = normalize(to_light);

                    // For light PDF
                    {
                        const float t_shadow = dist_to_light - 1e-3f;
                        // Trace shadow ray
                        const bool is_hit = traceShadowRay(
                            params.handle, si.p, light_dir, 1e-3f, t_shadow);

                        // Next event estimation
                        if (!hit_object) {
                            const Vec3f bsdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                                si.surface_info.callable_id.bsdf, &si, si.surface_info.data, light_dir);

                            const float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                                si.surface_info.callable_id.pdf, &si, si.surface_info.data, light_dir);

                            const float cos_theta = dot(-light_dir, li.n);

                            // MIS weight
                            const float weight = balanceHeuristic(li.pdf, bsdf_pdf * cos_theta / dist);

                            result += weight * li.emission * bsdf * throughput / li.pdf;
                        }
                    }

                    // Evaluate BSDF
                    {
                        // Importance sampling according to the BSDF
                        optixDirectCall<void, SurfaceInteraction*, void*>(
                            si.surface_info.callable_id.sample, &si, si.surface_info.data);
                        
                        // Evaluate BSDF
                        const Vec3f bsdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                            si.surface_info.callable_id.bsdf, &si, si.surface_info.data, si.wi);

                        float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                            si.surface_info.callable_id.pdf, &si, si.surface_info.data, si.wi);

                        const float light_pdf = optixDirectCall<float, const LightInfo&, const Vec3f&, const Vec3f&, LightInteraction&>(
                            light.pdf_id, light, si.p, light_dir, li);
                        
                        const float weight = balanceHeuristic(bsdf_pdf, light_pdf);
                        throughput *= weight * bsdf / bsdf_pdf;
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Light sampling
// ----------------------------------------------------------------------------
// Plane light sampling
extern "C" DEVICE void __direct_callable__sample_light_plane(
    const LightInfo& light, 
    const Vec3f& p, 
    LightInteraction& li, 
    uint32_t& seed)
{
    const auto* plane = (const Plane::Data*)light.shape_data;

    // Sample local point on the area emitter
    const float x = rnd(seed, plane->min.x(), plane->max.x());
    const float z = rnd(seed, plane->min.z(), plane->max.z());

    Vec3f rnd_p(x, 0.0f, z);
    rnd_p = light.objToWorld.pointMul(rnd_p);
    li.p = rnd_p;
    li.n = normalize(light.objToWorld.normalMul(Vec3f(0.0f, 1.0f, 0.0f)));
    li.uv = Vec2f(
        (x - plane->min.x()) / (plane->max.x() - plane->min.x()), 
        (z - plane->min.y()) / (plane->max.y() - plane->min.y()));
    
    // Calcluate area of the light source
    const Vec3f p0 = light.objToWorld.pointMul(Vec3f(plane->min.x(), 0.0f, plane->min.y()));
    const Vec3f p1 = light.objToWorld.pointMul(Vec3f(plane->max.x(), 0.0f, plane->min.y()));
    const Vec3f p2 = light.objToWorld.pointMul(Vec3f(plane->min.x(), 0.0f, plane->max.y()));
    li.area = length(cross(p1 - p0, p2 - p0));

    const Vec3f wi = rnd_p - p;
    const float t = length(wi);
    const cos_theta = fabs(dot(li.n, normalize(wi)));
    if (cos_theta < math::eps)
        li.pdf = 0.0f;
    else
        li.pdf = t * t / (li.area * cos_theta);

    // Emission from light source
    const auto* area_light = (const AreaEmitter::Data*)light.surface_info.data;
    float is_emitted = 1.0f;
    if (!area->twosided)
        is_emitted = (float)(dot(li.n, normalize(wi)) > 0.0f);
    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(
        area_light->texture.prg_id, li.uv, area_light->texture.data);
    li.emission = is_emitted * base * area_light->intensity;
}

// Triangle light sampling
static INLINE DEVICE Vec3f randomSampleOnTriangle(uint32_t& seed, const Triangle& triangle) {

    Vec2f uv = UniformSampler::get2D(seed);

    return barycentricInterop(triangle.v0, triangle.v1, triangle.v2, uv);
}

extern "C" DEVICE void __direct_callable__sample_light_triangle(
    const LightInfo& light,
    const Vec3f& p,
    LightInteraction& li,
    uint32_t& seed)
{
    const auto* triangle = (const Triangle::Data*)light.shape_data;

    // Sample local point on the light
    const Vec2f uv = UniformSampler::get2D(seed);
    li.p = randomSampleOnTriangle(seed, *triangle);
    li.n = normalize(triangle->n);
    li.uv = uv;
    li.area = 0.5f * length(cross(triangle->v1 - triangle->v0, triangle->v2 - triangle->v0));

    // PDF
    const Vec3f wi = li.p - p;
    Vec3f N = triangle->n;
    N = faceforward(N, -wi, N);
    const float t = length(wi);
    const float cos_theta = fabs(dot(N, normalize(wi)));
    if (cos_theta < math::eps)
        li.pdf = 0.0f;
    else
        li.pdf = t * t / (li.area * cos_theta);

    // Emission from light source
    const auto* area_light = (const AreaEmitter::Data*)light.surface_info.data;
    float is_emitted = 1.0f;
    if (!area->twosided)
        is_emitted = (float)(dot(li.n, normalize(wi)) > 0.0f);
    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(
        area_light->texture.prg_id, li.uv, area_light->texture.data);
    li.emission = is_emitted * base * area_light->intensity;
}