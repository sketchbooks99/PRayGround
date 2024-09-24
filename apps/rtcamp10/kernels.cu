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


            if (si.surface_info->type == SurfaceType::AreaEmitter) {
                // Evaluating emission from emitter
                Vec3f emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.sample, &si, si.surface_info->data
                );

                result += throughput * emission;
                if (si.trace_terminate)
                    break;
            } 
            // Specular surfaces
            else if (+(si.surface_info->type & SurfaceType::Delta)) {
                // Sample scattered ray
                auto wi = optixDirectCall<ScatteredRay, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.sample, &si, si.surface_info->data
                );
                // Both
                if (wi.scattered_type & 3)
                    si.wi = rnd(si.seed) < wi.reflect_prob ? wi.reflected : wi.transmitted;
                else if (wi.scattered_type & 1)
                    si.wi = wi.reflected;
                else if (wi.scattered_type & 2)
                    si.wi = wi.transmitted;

                // Evaluate BSDF
                Vec3f bsdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&>(
                    si.surface_info->callable_id.bsdf, &si, si.surface_info->data, si.wi, si.wo
                );

                throughput *= bsdf;
            }
            // Rough surface sampling with MIS
            else if (+(si.surface_info->type & SurfaceType::Rough)) {
                auto wi = optixDirectCall<ScatteredRay, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.sample, &si, si.surface_info->data
                );
                si.wi = wi.scattered_type & 1 ? wi.reflected : wi.transmitted;

                Vec3f bsdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&>(
                    si.surface_info->callable_id.bsdf, &si, si.surface_info->data, si.wi, si.wo
                );

                float pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&>(
                    si.surface_info->callable_id.pdf, &si, si.surface_info->data, si.wi, si.wo
                );

                throughput *= bsdf / pdf;
                //LightInfo light;
                //if (params.num_lights > 0) {
                //    const int light_id = rndInt(si.seed, 0, params.num_lights - 1);
                //    light = params.lights[light_id];
                //}

                //float pdf = 0.0f;

                //if (params.num_lights > 0) {
                //    LightInteraction li;
                //    // Sampling light point
                //    optixDirectCall<void, SurfaceInteraction*, LightInfo*, LightInteraction*, void*>(
                //        si.surface_info->callable_id.sample, &si, &light, &li, si.surface_info->data
                //    );
                //    Vec3f to_light = li.p - si.p;
                //    const float dist = length(to_light);
                //    const Vec3f light_dir = normalize(to_light);

                //    // For light PDF
                //    {
                //        const float t_shadow = dist_to_light - 1e-3f;
                //        // Trace shadow ray
                //        const bool is_hit = traceShadowRay(
                //            params.handle, si.p, light_dir, 1e-3f, t_shadow);

                //        // Next event estimation
                //        if (!hit_object) {
                //            const Vec3f bsdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                //                si.surface_info->callable_id.bsdf, &si, si.surface_info->data, light_dir);

                //            const float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                //                si.surface_info->callable_id.pdf, &si, si.surface_info->data, light_dir);

                //            const float cos_theta = dot(-light_dir, li.n);

                //            // MIS weight
                //            const float weight = balanceHeuristic(li.pdf, bsdf_pdf * cos_theta / dist);

                //            result += weight * li.emission * bsdf * throughput / li.pdf;
                //        }
                //    }

                //    // Evaluate BSDFSample
                //    {
                //        // Importance sampling according to the BSDFSample
                //        optixDirectCall<void, SurfaceInteraction*, void*>(
                //            si.surface_info->callable_id.sample, &si, si.surface_info->data);
                //        
                //        // Evaluate BSDFSample
                //        const Vec3f bsdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                //            si.surface_info->callable_id.bsdf, &si, si.surface_info->data, si.wi);

                //        float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                //            si.surface_info->callable_id.pdf, &si, si.surface_info->data, si.wi);

                //        const float light_pdf = optixDirectCall<float, const LightInfo&, const Vec3f&, const Vec3f&, LightInteraction&>(
                //            light.pdf_id, light, si.p, light_dir, li);
                //        
                //        const float weight = balanceHeuristic(bsdf_pdf, light_pdf);
                //        throughput *= weight * bsdf / bsdf_pdf;
                //    }
                //}
            }

            ro = si.p;
            rd = si.wi;

            ++depth;
        } // for (;;)
        i--;
    } // while (i > 0)

    if (!result.isValid()) result = 0.0f;

    Vec3f accum_color = result / static_cast<float>(params.samples_per_launch);

    if (frame > 0) {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_color_prev = params.accum_buffer[image_idx];
        accum_color = lerp(accum_color_prev, accum_color, a);
    }

    params.accum_buffer[image_idx] = Vec4f(accum_color, 1.0f);
    Vec3u color = make_color(accum_color);
    params.result_buffer[image_idx] = Vec4u(color, 255);
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

    si->shading.uv = shading.uv;
    si->trace_terminate = true;
    si->emission = optixDirectCall<Vec3f, const Vec2f&, void*>(env->texture.prg_id, si->shading.uv, env->texture.data);
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
        Frame shading_frame = Frame::FromXZ(shading.dpdu, shading.n);
        // Fetch bumpmap normal 
        Vec3f n = optixDirectCall<Vec3f, Vec2f&, void*>(data->surface_info->bumpmap.prg_id, shading.uv, data->surface_info->bumpmap.data);
        n = normalize(n * 2.0f - 1.0f);
        // Transform normal from tangent space to local space
        n = shading_frame.fromLocal(n);
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
extern "C" DEVICE ScatteredRay __direct_callable__sample_diffuse(SurfaceInteraction* si, void* data, const Vec3f& wo) {
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(data);

    si->wi = pgImportanceSamplingDiffuse(diffuse, si->wo, si->shading, si->seed);
    si->trace_terminate = false;

    return { si->wi, Vec3f(0.0f), 1.0f, 1};
}

extern "C" DEVICE Vec3f __direct_callable__bsdf_diffuse(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo) {
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(data);

    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(
        diffuse->texture.prg_id, si->shading.uv, diffuse->texture.data);
    si->albedo = albedo;
    si->emission = 0.0f;
    return albedo * pgGetDiffuseBRDF(wi, si->shading.n);
}

extern "C" DEVICE float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo) {
    return pgGetDiffusePDF(wi, si->shading.n);
}

// Specular reflection
extern "C" DEVICE ScatteredRay __direct_callable__sample_conductor(SurfaceInteraction* si, void* data, const Vec3f& wo) {
    const Conductor::Data* conductor = reinterpret_cast<Conductor::Data*>(data);

    if (conductor->twosided)
        si->shading.n = faceforward(si->shading.n, si->wo, si->shading.n);
    si->wi = reflect(-si->wo, si->shading.n);
    si->trace_terminate = false;
    return { si->wi, Vec3f(0.0f), 1.0f, 1};
}

extern "C" DEVICE Vec3f __direct_callable__bsdf_conductor(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo) {
    const Conductor::Data* conductor = reinterpret_cast<Conductor::Data*>(data);

    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(
        conductor->texture.prg_id, si->shading.uv, conductor->texture.data);
    si->albedo = albedo;
    // Apply thinfilm interaction
    const float cos_theta = dot(si->wo, si->shading.n);
    Vec3f tf_thickness = optixDirectCall<Vec3f, const Vec2f&, void*>(conductor->thinfilm.thickness.prg_id, si->shading.uv, conductor->thinfilm.thickness.data);
    tf_thickness *= conductor->thinfilm.thickness_scale;
    Vec3f thinfilm = fresnelAiry(1.0f, cos_theta, conductor->thinfilm.ior, conductor->thinfilm.extinction, tf_thickness.x(), conductor->thinfilm.tf_ior);

    return thinfilm * albedo;
}

extern "C" float __direct_callable__pdf_conductor(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo) {
    return 1.0f;
}

// Specular transmission
extern "C" DEVICE ScatteredRay __direct_callable__sample_dielectric(SurfaceInteraction* si, void* data, const Vec3f& wo) {
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(data);

    float ni = 1.000292f;       /// @todo Consider IOR of current medium where ray goes on
    float nt = dielectric->ior;
    Vec3f wo = -si->wo;
    float cosine = dot(wo, si->shading.n);
    // Check where the ray is going outside or inside
    bool into = cosine < 0;
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
    if (cannot_refract)
        return { reflect(wo, outward_normal), Vec3f(0.0f), 1.0f, 1 };
    else
        return { reflect(wo, outward_normal), refract(wo, outward_normal, cosine, ni, nt), reflect_prob, 1 | 2 };
}

extern "C" DEVICE Vec3f __direct_callable__bsdf_dielectric(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo) {
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(data);

    bool into = dot(wo, si->shading.n) > 0.0f;

    // Evaluate BSDFSample
    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(
        dielectric->texture.prg_id, si->shading.uv, dielectric->texture.data);
    si->albedo = albedo;
    si->emission = 0.0f;
    const float cos_theta = dot(wo, si->shading.n);
    float ni = 1.0f;
    float nt = dielectric->ior;
    if (!into)
        swap(ni, nt);

    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    bool cannot_refract = (ni / nt) * sin_theta > 1.0f;

    Vec3f tf_thickness = optixDirectCall<Vec3f, const Vec2f&, void*>(dielectric->thinfilm.thickness.prg_id, si->shading.uv, dielectric->thinfilm.thickness.data);
    Vec3f tf_value = fresnelAiry(1.0f, cos_theta, dielectric->ior, dielectric->thinfilm.extinction, tf_thickness.x(), dielectric->thinfilm.tf_ior) * albedo;

    Vec3f bsdf = albedo;
    if (into)
        bsdf *= tf_value;
    return bsdf;
}

extern "C" DEVICE float __direct_callable__pdf_dielectric(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo) {
    return 1.0f;
}

// Disney
extern "C" DEVICE ScatteredRay __direct_callable__sample_disney(SurfaceInteraction* si, void* data, const Vec3f& wo) {
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(data);

    // Importance sampling
    si->wi = pgImportanceSamplingDisney(disney, -si->wo, si->shading, si->seed);
    si->trace_terminate = false;
    return { si->wi, Vec3f(0.0f), 1.0f, 1 };
}

extern "C" DEVICE Vec3f __direct_callable__bsdf_disney(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo) {
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(data);

    // Evaluate BSDF
    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(disney->albedo.prg_id, si->shading.uv, disney->albedo.data);
    si->albedo = albedo;
    const float cos_theta = dot(si->wi, si->shading.n);
    Vec3f tf_thickness = optixDirectCall<Vec3f, const Vec2f&, void*>(disney->thinfilm.thickness.prg_id, si->shading.uv, disney->thinfilm.thickness.data);
    tf_thickness *= disney->thinfilm.thickness_scale;

    Vec3f tf_value = fresnelAiry(1.0f, cos_theta, disney->thinfilm.ior, disney->thinfilm.extinction, tf_thickness.x(), disney->thinfilm.tf_ior);
    float mag_albedo = length(albedo);
    tf_value = normalize(tf_value) * mag_albedo;
    Vec3f bsdf = pgGetDisneyBRDF(disney, -si->wo, si->wi, si->shading, tf_value);
    return bsdf;
}

extern "C" DEVICE float __direct_callable__pdf_disney(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo) {
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(data);

    // Evaluate PDF
    return pgGetDisneyPDF(disney, -si->wo, wi, si->shading);
}

// Layered material
extern "C" DEVICE ScatteredRay __direct_callable__sample_layered(SurfaceInteraction* si, void* data, const Vec3f& wo) {
    const Layered::Data* layered = reinterpret_cast<Layered::Data*>(data);

    uint32_t n_layers = layered->num_layers;
    int32_t l = 0;

    Vec3f wi;
    Vec3f _wo = wo;
    float sample_prob;
    int32_t l = 0;

    // Downward ray tracing into the layered surface
    while (l < n_layers) {
        SurfaceInfo info = si->surface_info[l];
        // Generate refraction and reflection ray and increment layer if interacting surface is refractive
        if (+(info.type & SurfaceType::Refractive)) {
            ScatteredRay scattered = optixDirectCall<ScatteredRay, SurfaceInteraction*, void*>(info.callable_id.sample, si, info.data);
            // If refractive surface cannot refract, generate reflection ray and don't increment layer
            if (rnd(si->seed) < scattered.reflect_prob)  {
                wi = scattered.reflected;
                break;
            }
            // Transmit the ray to the next layer
            else {
                _wo = -scattered.transmitted;
                ++l;
            }
        }
        // Generate reflection ray and don't increment layer
        else {
            ScatteredRay scattered = optixDirectCall<ScatteredRay, SurfaceInteraction*, void*>(info.callable_id.sample, si, info.data);
            wi = scattered.reflected;
            break;
        }
    }

    // Upward ray tracing from the bottom layer to the top layer
    for (int32_t i = l - 1; i >= 0; --i) {
        SurfaceInfo info = si->surface_info[i];
        ScatteredRay scattered = optixDirectCall<ScatteredRay, SurfaceInteraction*, void*>(
            info.callable_id.bsdf, si, info.data, wi);

        // Ray cannot go through from the surface to the outside
        if (rnd(si->seed) < scattered.reflect_prob) {
            si->trace_terminate = true;
        }
        else {
            wi = scattered.transmitted;
        }
    }

    return { wi, Vec3f(0.0f), 1.0f, 1 };
}

extern "C" DEVICE Vec3f __direct_callable__bsdf_layered(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo) {
    const Layered::Data* layered = reinterpret_cast<Layered::Data*>(data);

    uint32_t n_layers = layered->num_layers;

    int32_t l = 0;
    Vec3f total_bsdf;
    while (l < n_layers) {
        SurfaceInfo info = si->surface_info[l];
        Vec3f bsdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&, const Vec3f&>(
            info.callable_id.bsdf, si, info.data, wi, wo);
        ++l;
    }
}

extern "C" DEVICE float __direct_callable__pdf_layered(SurfaceInteraction* si, void* data, const Vec3f& wi, const Vec3f& wo) {
    const Layered::Data* layered = reinterpret_cast<Layered::Data*>(data);

    uint32_t n_layers = layered->num_layers;
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