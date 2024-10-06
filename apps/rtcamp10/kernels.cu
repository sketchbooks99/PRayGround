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
};

struct BSDFSample {
    Vec3f value;
    float pdf;
    Vec3f wi;
};

struct BSDFProperty {
    Vec3f bsdf;
    float thickness;
};

enum SCATTERED_TYPE {
    REFLECTED = 1,
    TRANSMITTED = 2,
    BOTH = 3
};

struct ScatteredRay {
    Vec3f reflected;
    Vec3f transmitted;
    float reflect_prob;
    // 1: Reflected, 2: Transmitted, 1 | 2: Both
    uint8_t scattered_type; 
};

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
    const pgRaygenData<Camera>* rg = reinterpret_cast<pgRaygenData<Camera>*>(optixGetSbtDataPointer());

    const int frame = params.frame;

    const Vec3ui idx(optixGetLaunchIndex());

    const int image_idx = idx.y() * params.width + idx.x();
    uint32_t seed = tea<4>(image_idx, frame);

    Vec3f result(0.0f);
    Vec3f albedo(0.0f);
    Vec3f normal(0.0f);

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
        SurfaceInfo surface_info;
        surface_info.type == SurfaceType::None;
        si.surface_info = &surface_info;

        int depth = 0;
        for (;;) {
            if (depth >= params.max_depth)
                break;

            trace(params.handle, ro, rd, 1e-3f, 1e10f, &si);

            if (si.trace_terminate) {
                result += throughput * si.emission;
                break;
            }

            SurfaceInfo surface_info = si.surface_info[0];

            if (surface_info.type == SurfaceType::AreaEmitter) {
                // Evaluating emission from emitter
                Vec3f emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
                    surface_info.callable_id.sample, &si, surface_info.data
                );

                result += throughput * emission;
                if (si.trace_terminate)
                    break;
            } 
            // Specular surfaces
            else if (+(surface_info.type & SurfaceType::Delta)) {
                // Sample scattered ray
                ScatteredRay out_ray; 
                float pdf = 1.0f;
                optixDirectCall<void, SurfaceInteraction*, void*, const Vec3f&, ScatteredRay&, float&>(
                    surface_info.callable_id.sample, &si, surface_info.data, si.wo, out_ray, pdf
                );
                if (out_ray.scattered_type == REFLECTED)
                    si.wi = out_ray.reflected;
                else if (out_ray.scattered_type == TRANSMITTED)
                    si.wi = out_ray.transmitted;

                // Evaluate BSDF
                BSDFProperty bsdf = optixDirectCall<BSDFProperty, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&, uint8_t>(
                    surface_info.callable_id.bsdf, &si, surface_info.data, si.wi, si.wo, out_ray.scattered_type
                );

                if (!bsdf.bsdf.isValid())
                    break;

                throughput *= bsdf.bsdf;
            }
            // Rough surface sampling with MIS
            else if (+(surface_info.type & (SurfaceType::Rough | SurfaceType::Layered))) {
                //ScatteredRay out_ray;
                //float pdf = 0.0f;
                //optixDirectCall<void, SurfaceInteraction*, void*, const Vec3f&, ScatteredRay&, float&>(
                //    surface_info.callable_id.sample, &si, surface_info.data, si.wo, out_ray, pdf
                //);
                //si.wi = out_ray.reflected;
                //BSDFProperty bsdf = optixDirectCall<BSDFProperty, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&, uint8_t>(
                //    surface_info.callable_id.bsdf, &si, surface_info.data, si.wi, si.wo, out_ray.scattered_type
                //);

                //throughput *= bsdf.bsdf / pdf;
                LightInfo light;
                if (params.num_lights > 0) {
                    const int light_id = rndInt(si.seed, 0, params.num_lights - 1);
                    light = params.lights[light_id];

                    LightInteraction li;
                    // Sampling light point
                    optixDirectCall<void, const LightInfo&, const Vec3f&, LightInteraction&, uint32_t&>(
                        light.sample_id, light, si.p, li, si.seed
                    );
                    Vec3f to_light = li.p - si.p;
                    const float dist = length(to_light);
                    const Vec3f light_dir = normalize(to_light);

                    // For light PDF
                    {
                        const float t_shadow = dist - 1e-3f;
                        // Trace shadow ray
                        const bool is_hit = traceShadowRay(
                            params.handle, si.p, light_dir, 1e-3f, t_shadow);

                        // Next event estimation
                        if (!is_hit) {
                            BSDFProperty bsdf = optixDirectCall<BSDFProperty, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&, uint8_t>(
                                surface_info.callable_id.bsdf, &si, surface_info.data, light_dir, si.wo, REFLECTED);

                            const float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&, uint8_t>(
                                surface_info.callable_id.pdf, &si, surface_info.data, light_dir, si.wo, REFLECTED);

                            const float cos_theta = dot(-light_dir, li.n);

                            li.pdf /= params.num_lights;

                            // MIS weight
                            const float weight = balanceHeuristic(li.pdf, bsdf_pdf * cos_theta / dist);

                            result += weight * li.emission * bsdf.bsdf * throughput / li.pdf;
                        }
                    }

                    // Evaluate BSDFSample
                    {
                        // Importance sampling according to the BSDFSample
                        ScatteredRay out_ray;
                        float bsdf_pdf = 0.0f;
                        optixDirectCall<void, SurfaceInteraction*, void*, const Vec3f&, ScatteredRay&, float&>(
                            surface_info.callable_id.sample, &si, surface_info.data, si.wo, out_ray, bsdf_pdf
                        );
                        si.wi = out_ray.reflected;
                        BSDFProperty bsdf = optixDirectCall<BSDFProperty, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&, uint8_t>(
                            surface_info.callable_id.bsdf, &si, surface_info.data, si.wi, si.wo, out_ray.scattered_type
                        );

                        const float light_pdf = optixDirectCall<float, const LightInfo&, const Vec3f&, const Vec3f&>(
                            light.pdf_id, light, si.p, light_dir);
                        
                        const float weight = balanceHeuristic(bsdf_pdf, light_pdf);
                        throughput *= weight * bsdf.bsdf / bsdf_pdf;
                    }
                }
            }

            if (depth == 0) {
                albedo += si.albedo;
                normal += si.shading.n;
            }

            ro = si.p;
            rd = si.wi;

            ++depth;
        } // for (;;)
        i--;
    } // while (i > 0)

    if (!result.isValid()) result = 0.0f;

    Vec3f accum_color = result / static_cast<float>(params.samples_per_launch);
    Vec3f accum_normal = normal / static_cast<float>(params.samples_per_launch);
    Vec3f accum_albedo = albedo / static_cast<float>(params.samples_per_launch);

    if (frame > 0) {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_color_prev = params.accum_buffer[image_idx];
        accum_color = lerp(accum_color_prev, accum_color, a);
        const Vec3f albedo_prev(params.albedo_buffer[image_idx]);
        const Vec3f normal_prev(params.normal_buffer[image_idx]);
        accum_albedo = lerp(albedo_prev, accum_albedo, a);
        accum_normal = lerp(normal_prev, accum_normal, a);
    }

    params.accum_buffer[image_idx] = Vec4f(accum_color, 1.0f);
    Vec3u color = make_color(accum_color);
    params.result_buffer[image_idx] = Vec4f(color2float(color), 1.0f);
    params.normal_buffer[image_idx] = Vec4f(accum_normal, 1.0f);
    params.albedo_buffer[image_idx] = Vec4f(accum_albedo, 1.0f);
}

// ----------------------------------------------------------------------------
// Miss program
// ----------------------------------------------------------------------------
extern "C" DEVICE void __miss__envmap() {
    pgMissData* data = reinterpret_cast<pgMissData*>(optixGetSbtDataPointer());
    auto* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    Ray ray = getWorldRay();

    Shading shading;
    float t;
    const Sphere::Data env_sphere{ Vec3f(0.0f), 1e8f };
    pgIntersectionSphere(&env_sphere, ray, &shading, &t);

    Vec3f color = optixDirectCall<Vec3f, const Vec2f&, void*>(env->texture.prg_id, shading.uv, env->texture.data);

    si->shading.uv = shading.uv;
    si->trace_terminate = true;
    si->emission = color;
    si->albedo = color;
    si->shading.n = shading.n;
}

extern "C" DEVICE void __miss__shadow() {
    setPayload<0>(0u);
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
    const float z = rnd(seed, plane->min.y(), plane->max.y());

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
    const float cos_theta = fabs(dot(li.n, normalize(wi)));
    if (cos_theta < math::eps)
        li.pdf = 0.0f;
    else
        li.pdf = t * t / (li.area * cos_theta);

    // Emission from light source
    const auto* area_light = (const AreaEmitter::Data*)light.surface_info->data;
    float is_emitted = 1.0f;
    if (!area_light->twosided)
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
    const auto* triangle = (const Triangle*)light.shape_data;

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
    const auto* area_light = (const AreaEmitter::Data*)light.surface_info->data;
    float is_emitted = 1.0f;
    if (!area_light->twosided)
        is_emitted = (float)(dot(li.n, normalize(wi)) > 0.0f);
    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(
        area_light->texture.prg_id, li.uv, area_light->texture.data);
    li.emission = is_emitted * base * area_light->intensity;
}


// Sphere emitter
extern "C" __device__ void __direct_callable__sample_light_sphere(
    const LightInfo& light, const Vec3f& p, LightInteraction& li, uint32_t& seed)
{
    const auto* sphere = (Sphere::Data*)light.shape_data;
    const Vec3f center = sphere->center;
    const Vec3f local_p = light.worldToObj.pointMul(p);
    const Vec3f oc = center - local_p;
    float distance_squared = dot(oc, oc);
    Onb onb(normalize(oc));
    Vec3f to_light = randomSampleToSphere(seed, sphere->radius, distance_squared);
    onb.inverseTransform(to_light);
    li.p = light.objToWorld.pointMul(center + to_light);
    
    Shading shading;
    Ray ray(local_p, to_light, 1e-3f, 1e10f);
    float t = 0.0f;
    bool is_hit = pgIntersectionSphere(sphere, ray, &shading, &t);
    li.n = normalize(shading.n);
    li.uv = shading.uv;
    const float cos_theta_max = sqrtf(1.0f - sphere->radius * sphere->radius / pow2(length(center - local_p)));
    const float solid_angle = 2.0f * math::pi * (1.0f - cos_theta_max);
    li.area = solid_angle * pow2(sphere->radius) * 4.0f * math::pi;

    const float cos_theta = fabs(dot(li.n, normalize(-to_light)));
    li.pdf = t * t / (li.area * cos_theta);

    const auto* area_light = (const AreaEmitter::Data*)light.surface_info->data;
    float is_emitted = 1.0f;
    if (!area_light->twosided)
        is_emitted = (float)(dot(li.n, normalize(-to_light)) > 0.0f);
    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(
        area_light->texture.prg_id, li.uv, area_light->texture.data);
    li.emission = is_emitted * base * area_light->intensity;
}

extern "C" __device__ float __direct_callable__pdf_light_sphere(
    const LightInfo& light, const Vec3f& p, const Vec3f& wi)
{
    const auto* sphere = (Sphere::Data*)light.shape_data;
    const Vec3f local_p = light.worldToObj.pointMul(p);
    const Vec3f local_wi = light.worldToObj.vectorMul(wi);

    Shading shading{};
    Ray ray(local_p, local_wi, 1e-3f, 1e10f);
    float time = 0.0f;
    if (!pgIntersectionSphere(sphere, ray, &shading, &time))
        return 0.0f;

    const Vec3f center = sphere->center;
    const float radius = sphere->radius;
    const float cos_theta_max = sqrtf(1.0f - radius * radius / pow2(length(center - local_p)));
    const float solid_angle = math::two_pi * (1.0f - cos_theta_max);
    return 1.0f / solid_angle;
}

// ----------------------------------------------------------------------------
// Hitgroups
// ----------------------------------------------------------------------------
// Sphere
extern "C" DEVICE void __intersection__sphere() {
    const pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

    Ray ray = getLocalRay();
    pgReportIntersectionSphere(sphere, ray);
}

extern "C" DEVICE void __intersection__plane() {
    const pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const Plane::Data* plane = reinterpret_cast<Plane::Data*>(data->shape_data);

    Ray ray = getLocalRay();
    pgReportIntersectionPlane(plane, ray);
}

extern "C" DEVICE void __closesthit__custom() {
    const pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    Shading* shading = getPtrFromTwoAttributes<Shading, 0>();

    //  Transform shading from object to world space
    shading->n = normalize(optixTransformNormalFromObjectToWorldSpace(shading->n));
    shading->dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(shading->dpdu));
    shading->dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(shading->dpdv));

    auto* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    si->p = ray.at(ray.tmax);
    si->shading = *shading;
    si->t = ray.tmax;
    si->wo = -ray.d;
    si->surface_info = data->surface_info;
}

// Mesh
extern "C" DEVICE void __closesthit__mesh() {
    const pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const TriangleMesh::Data* mesh_data = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Shading shading = pgGetMeshShading(mesh_data, optixGetTriangleBarycentrics(), optixGetPrimitiveIndex());

    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    if (data->surface_info->use_bumpmap) {
        Vec3f n = optixDirectCall<Vec3f, Vec2f&, void*>(data->surface_info->bumpmap.prg_id, shading.uv, data->surface_info->bumpmap.data);
        n = normalize(n * 2.0f - 1.0f);
        Onb onb(shading.n);
        onb.inverseTransform(n);
        shading.n = normalize(n);
    }

    // Transform shading from object to world space
    shading.n = normalize(optixTransformNormalFromObjectToWorldSpace(shading.n));
    shading.dpdu = optixTransformVectorFromObjectToWorldSpace(shading.dpdu);
    shading.dpdv = optixTransformVectorFromObjectToWorldSpace(shading.dpdv);

    si->p = ray.at(ray.tmax);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = -ray.d;
    si->surface_info = data->surface_info;
}

extern "C" DEVICE void __closesthit__shadow() {
    setPayload<0>(1u);
}

// ----------------------------------------------------------------------------
// Surface 
// ----------------------------------------------------------------------------
// Diffuse
extern "C" DEVICE void __direct_callable__sample_diffuse(SurfaceInteraction* si, void* data, const Vec3f& wo, ScatteredRay& out_ray, float& pdf) {
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(data);
    si->wi = pgImportanceSamplingDiffuse(diffuse, wo, si->shading, si->seed);
    si->trace_terminate = false;

    out_ray = ScatteredRay{ si->wi, Vec3f(0.0f), 1.0f, REFLECTED};
    pdf = pgGetDiffusePDF(si->wi, si->shading.n);
}

extern "C" DEVICE BSDFProperty __direct_callable__bsdf_diffuse(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo, uint8_t scatter_type) {
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(data);

    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(
        diffuse->texture.prg_id, si->shading.uv, diffuse->texture.data);
    si->albedo = albedo;
    si->emission = 0.0f;

    return BSDFProperty{ albedo * pgGetDiffuseBRDF(wi, si->shading.n), 0.0f };
}

extern "C" DEVICE float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo) {
    return pgGetDiffusePDF(wi, si->shading.n);
}

// Specular reflection
extern "C" DEVICE void __direct_callable__sample_conductor(SurfaceInteraction* si, void* data, const Vec3f& wo, ScatteredRay& out_ray, float& pdf) {
    const Conductor::Data* conductor = reinterpret_cast<Conductor::Data*>(data);

    if (conductor->twosided)
        si->shading.n = faceforward(si->shading.n, wo, si->shading.n);
    si->wi = reflect(-wo, si->shading.n);
    si->trace_terminate = false;
    out_ray = ScatteredRay{ si->wi, Vec3f(0.0f), 1.0f, REFLECTED};
    pdf = 1.0f;
}

extern "C" DEVICE BSDFProperty __direct_callable__bsdf_conductor(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo, uint8_t scatter_type) {
    const Conductor::Data* conductor = reinterpret_cast<Conductor::Data*>(data);

    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(
        conductor->texture.prg_id, si->shading.uv, conductor->texture.data);
    si->albedo = albedo;

    if (conductor->thinfilm.thickness.prg_id == -1)
        return BSDFProperty{ albedo, 0.0f };

    // Apply thinfilm interaction
    const float cos_theta = dot(wi, si->shading.n);
    Vec3f tf_thickness = optixDirectCall<Vec3f, const Vec2f&, void*>(conductor->thinfilm.thickness.prg_id, si->shading.uv, conductor->thinfilm.thickness.data);
    tf_thickness *= conductor->thinfilm.thickness_scale;
    Vec3f thinfilm = fresnelAiry(1.0f, cos_theta, conductor->thinfilm.ior, conductor->thinfilm.extinction, tf_thickness.x(), conductor->thinfilm.tf_ior);

    si->albedo = albedo * thinfilm;

    return BSDFProperty{ si->albedo, tf_thickness.x() };
}

extern "C" DEVICE float __direct_callable__pdf_conductor(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo, uint8_t scatter_type) {
    return 1.0f;
}

// Specular transmission
extern "C" DEVICE void __direct_callable__sample_dielectric(SurfaceInteraction* si, void* data, const Vec3f& wo, ScatteredRay& out_ray, float& pdf) {
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(data);

    float ni = 1.000292f;       /// @todo Consider IOR of current medium where ray goes on
    float nt = dielectric->ior;
    float cosine = dot(wo, si->shading.n);
    // Check where the ray is going outside or inside
    bool into = cosine > 0;
    Vec3f outward_normal = into ? si->shading.n : -si->shading.n;

    // Swap IOR based on ray location
    if (!into) swap(ni, nt);

    // Check if the ray can be refracted
    cosine = fabs(cosine);
    float sine = sqrtf(1.0f - pow2(cosine));
    bool cannot_refract = (ni / nt) * sine > 1.0f;

    // Get reflectivity by the Fresnel equation
    float reflect_prob = fresnel(cosine, ni, nt);
    // Get out going direction of the ray
    if (cannot_refract) {
        out_ray = ScatteredRay{ reflect(-wo, outward_normal), Vec3f(0.0f), 1.0f, REFLECTED };
        pdf = 1.0f;
    }
    else {
        uint8_t scattered_type = rnd(si->seed) < reflect_prob ? REFLECTED : TRANSMITTED;
        out_ray = ScatteredRay { reflect(-wo, outward_normal), refract(-wo, outward_normal, cosine, ni, nt), reflect_prob, scattered_type };
        pdf = 1.0f;
    }
}

extern "C" DEVICE BSDFProperty __direct_callable__bsdf_dielectric(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo, uint8_t scatter_type) {
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(data);

    // Evaluate BSDFSample
    Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(
        dielectric->texture.prg_id, si->shading.uv, dielectric->texture.data);
    si->albedo = albedo;

    float ni = 1.000292f;       /// @todo Consider IOR of current medium where ray goes on
    float nt = dielectric->ior;
    float cosine = dot(wo, si->shading.n);
    // Check where the ray is going outside or inside
    bool into = cosine > 0;
    Vec3f outward_normal = into ? si->shading.n : -si->shading.n;

    // Swap IOR based on ray location
    if (!into) swap(ni, nt);

    // Check if the ray can be refracted
    cosine = fabs(cosine);
    float sine = sqrtf(1.0f - pow2(cosine));
    bool cannot_refract = (ni / nt) * sine > 1.0f;

    // Get reflectivity by the Fresnel equation
    float reflect_prob = fresnel(cosine, ni, nt);

    if (dielectric->thinfilm.thickness.prg_id == -1)
        return BSDFProperty{ albedo, 0.0f };

    Vec3f tf_thickness = optixDirectCall<Vec3f, const Vec2f&, void*>(dielectric->thinfilm.thickness.prg_id, si->shading.uv, dielectric->thinfilm.thickness.data);
    tf_thickness *= dielectric->thinfilm.thickness_scale;
    Vec3f tf_value = fresnelAiry(1.0f, cosine, dielectric->ior, dielectric->thinfilm.extinction, tf_thickness.x(), dielectric->thinfilm.tf_ior);
    tf_value *= tf_thickness.x() / dielectric->thinfilm.thickness_scale;

    if (into)
        albedo *= tf_value;
    si->albedo = albedo;
    return BSDFProperty{ albedo, tf_thickness.x() };
}

extern "C" DEVICE float __direct_callable__pdf_dielectric(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo, uint8_t scatter_type) {
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(data);

    const float cos_theta = dot(wo, si->shading.n);
    bool into = cos_theta > 0.0f;

    // Evaluate BSDFSample
    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(
        dielectric->texture.prg_id, si->shading.uv, dielectric->texture.data);
    si->albedo = albedo;
    si->emission = 0.0f;
    float ni = 1.0f;
    float nt = dielectric->ior;
    if (!into)
        swap(ni, nt);

    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    bool cannot_refract = (ni / nt) * sin_theta > 1.0f;
    float reflect_prob = fresnel(cos_theta, ni, nt);
    float pdf = 1.0f;
    if (!cannot_refract)
        pdf = scatter_type == REFLECTED ? reflect_prob : 1.0f - reflect_prob;
    return pdf;
}

// Disney
extern "C" DEVICE void __direct_callable__sample_disney(SurfaceInteraction* si, void* data, const Vec3f& wo, ScatteredRay& out_ray, float& pdf) {
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(data);

    // Importance sampling
    si->wi = pgImportanceSamplingDisney(disney, wo, si->shading, si->seed);
    si->trace_terminate = false;
    out_ray = ScatteredRay{ si->wi, Vec3f(0.0f), 1.0f, REFLECTED };
    pdf = pgGetDisneyPDF(disney, wo, si->wi, si->shading);
}

extern "C" DEVICE BSDFProperty __direct_callable__bsdf_disney(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo, uint8_t scatter_type) {
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(data);

    // Evaluate BSDF
    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(disney->albedo.prg_id, si->shading.uv, disney->albedo.data);
    si->albedo = albedo;

    if (disney->thinfilm.thickness.prg_id == -1) {
        return BSDFProperty{ pgGetDisneyBRDF(disney, wo, wi, si->shading, albedo), 0.0f };
    }

    const float cos_theta = dot(wi, si->shading.n);
    Vec3f tf_thickness = optixDirectCall<Vec3f, const Vec2f&, void*>(disney->thinfilm.thickness.prg_id, si->shading.uv, disney->thinfilm.thickness.data);
    tf_thickness *= disney->thinfilm.thickness_scale;

    Vec3f tf_value = fresnelAiry(1.0f, cos_theta, disney->thinfilm.ior, disney->thinfilm.extinction, tf_thickness.x(), disney->thinfilm.tf_ior);
    Vec3f bsdf = pgGetDisneyBRDF(disney, wo, wi, si->shading, albedo) * tf_value;
    return BSDFProperty{ bsdf, tf_thickness.x() };
}

extern "C" DEVICE float __direct_callable__pdf_disney(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo, uint8_t scatter_type) {
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(data);

    // Evaluate PDF
    return pgGetDisneyPDF(disney, -si->wo, wi, si->shading);
}

// Layered material
extern "C" DEVICE void __direct_callable__sample_layered(SurfaceInteraction* si, void* data, const Vec3f& wo, ScatteredRay& out_ray, float& pdf) {
    const Layered::Data* layered = reinterpret_cast<Layered::Data*>(data);

    uint32_t n_layers = layered->num_layers + 1;
    si->trace_terminate = false;

    Vec3f _wo = wo;
    Vec3f _wi;
    int32_t layer = 1;
    float pdf_weight = 0.5f;
    float total_pdf = 0.0f;

    // Downward ray sampling in the layered surface
    while (layer < n_layers) {
        SurfaceInfo info = si->surface_info[layer];
        ScatteredRay _out_ray;
        // Generate refraction and reflection ray and increment layer if interacting surface is refractive
        float _pdf = 0.0f;
        if (+(info.type & SurfaceType::Refractive)) {
            BSDFProperty bsdf = optixDirectCall<BSDFProperty, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&, uint8_t>(
                info.callable_id.bsdf, si, info.data, si->wi, _wo, 1
            );

            // Sample ray and get PDF
            optixDirectCall<void, SurfaceInteraction*, void*, const Vec3f&, ScatteredRay*, float*>
                (info.callable_id.sample, si, info.data, _wo, &_out_ray, &_pdf);

            if (_out_ray.scattered_type == REFLECTED) {
                _wi = _out_ray.reflected;
                total_pdf += _pdf * pdf_weight;
                break;
            } else {
                _wo = -_out_ray.transmitted;
                total_pdf += _pdf * pdf_weight;
                ++layer;
            }
        } else {
            optixDirectCall<void, SurfaceInteraction*, void*, const Vec3f&, ScatteredRay*, float*>
                (info.callable_id.sample, si, info.data, _wo, &_out_ray, &_pdf);
            _wi = _out_ray.reflected;

            total_pdf += _pdf * pdf_weight;
            break;
        }
        pdf_weight = 1.0f - pdf_weight;
    }

    // Upward ray sampling from the bottom layer to the top layer
    for (int32_t i = layer - 1; i >= 1; --i) {
        SurfaceInfo info = si->surface_info[i];
        ScatteredRay _out_ray;
        float _pdf = 0.0f;

        BSDFProperty bsdf = optixDirectCall<BSDFProperty, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&, uint8_t>(
            info.callable_id.bsdf, si, info.data, _wi, _wo, 1
        );

        if (bsdf.thickness <= 0.0f) {
            break;
        }

        optixDirectCall<void, SurfaceInteraction*, void*, const Vec3f&, ScatteredRay*, float*>
            (info.callable_id.sample, si, info.data, -_wi, &_out_ray, &_pdf);

        if (_out_ray.reflect_prob >= 1.0f) {
            //si->trace_terminate = true;
            break;
        } else {
            _wi = _out_ray.transmitted;
        }
    }
    
    out_ray = { _wi, Vec3f(0.0f), 1.0f, 1 };
    pdf = total_pdf;
}

extern "C" DEVICE BSDFProperty __direct_callable__bsdf_layered(SurfaceInteraction* si, void* data, const Vec3f& wo, const Vec3f& wi, uint8_t scatter_type) {
    const Layered::Data* layered = reinterpret_cast<Layered::Data*>(data);

    uint32_t n_layers = layered->num_layers + 1;

    int32_t layer = 1;
    Vec3f total_bsdf(0.0f);
    Vec3f t_wo = wo, t_wi = wi;
    float attenuation = 1.0f;
    Vec3f albedo(0.0f);
    while (layer < n_layers) {
        SurfaceInfo info = si->surface_info[layer];

        BSDFProperty bsdf = optixDirectCall<BSDFProperty, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&, uint8_t>(
            info.callable_id.bsdf, si, info.data, t_wi, t_wo, scatter_type
        );
        albedo += si->albedo * attenuation;

        total_bsdf = lerp(total_bsdf, bsdf.bsdf, attenuation);

        // Terminate recursive BSDF evaluation through the layered surface
        if (layer == n_layers - 1)
            break;

        // Transmitted ray into the next layer
        ScatteredRay out_ray;
        float pdf = 1.0f;
        optixDirectCall<void, SurfaceInteraction*, void*, const Vec3f&, ScatteredRay*, float*>(
            info.callable_id.sample, si, info.data, t_wo, &out_ray, &pdf);
        if (out_ray.reflect_prob >= 1.0f)
            break;
        t_wo = -out_ray.transmitted;
        optixDirectCall<void, SurfaceInteraction*, void*, const Vec3f&, ScatteredRay*, float*>(
            info.callable_id.sample, si, info.data, t_wi, &out_ray, &pdf);
        if (out_ray.reflect_prob >= 1.0f)
            break;
        t_wi = -out_ray.transmitted;

        const float cos_o = fabs(dot(t_wo, si->shading.n));
        const float cos_i = fabs(dot(t_wi, si->shading.n));

        float l = bsdf.thickness * 1e-1f * ((1.0f / cos_i) + (1.0f / cos_o));
        attenuation *= expf(-l);

        ++layer;
    }
    si->albedo = albedo;
    return BSDFProperty{ total_bsdf, 0.0f };
}

extern "C" DEVICE float __direct_callable__pdf_layered(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo, uint8_t scatter_type) {
    const Layered::Data* layered = reinterpret_cast<Layered::Data*>(data);

    uint32_t n_layers = layered->num_layers + 1;

    float total_pdf = 0.0f;
    int32_t layer = 1;
    float weight = 1.0f / (float)(n_layers - 1);
    while (layer < n_layers) {
        SurfaceInfo info = si->surface_info[layer];

        float pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&, uint8_t>(
            info.callable_id.pdf, si, info.data, wi, wo, scatter_type);

        total_pdf += weight * pdf;
        ++layer;
        weight = 1.0f - weight;
    }
    return total_pdf;
}

// Area emitter
extern "C" DEVICE Vec3f __direct_callable__area_emitter(SurfaceInteraction* si, void* data) {
    const AreaEmitter::Data* area = reinterpret_cast<AreaEmitter::Data*>(data);

    si->trace_terminate = true;
    float is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    if (area->twosided) {
        is_emitted = 1.0f;

        si->shading.n = faceforward(si->shading.n, si->wo, si->shading.n);
    }

    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(
        area->texture.prg_id, si->shading.uv, area->texture.data);
    si->albedo = base;
    si->emission = base * area->intensity * is_emitted;
    
    return si->emission;
}

// Textures
extern "C" DEVICE Vec3f __direct_callable__bitmap(const Vec2f& uv, void* data) {
    return pgGetBitmapTextureValue<Vec3f>(uv, data);
}

extern "C" DEVICE Vec3f __direct_callable__constant(const Vec2f& uv, void* data) {
    return pgGetConstantTextureValue<Vec3f>(uv, data);
}

extern "C" DEVICE Vec3f __direct_callable__checker(const Vec2f& uv, void* data) {
    return pgGetCheckerTextureValue<Vec3f>(uv, data);
}