#include <prayground/prayground.h>
#include "params.h"

// Utilities

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>( unpackPointer(u0, u1) ); 
}

static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd,
    float tmin, float tmax, uint32_t ray_type, SurfaceInteraction* si
)
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle, ro, rd, tmin, tmax, 0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE, ray_type, 2, ray_type,
        u0, u1
    );
}

static INLINE DEVICE bool shadowTrace(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd,
    float tmin, float tmax)
{
    uint32_t hit = 0u;
    optixTrace(handle, ro, rd,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(1), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        (uint32_t)RayType::SHADOW, (uint32_t)RayType::N_RAY, (uint32_t)RayType::SHADOW,
        hit);
    return static_cast<bool>(hit);
}

__device__ inline void copySurfaceInfo(SurfaceInfo* dst, const SurfaceInfo* src) {
    dst->data = src->data;
    dst->callable_id = src->callable_id;
    dst->type = src->type;
    dst->use_bumpmap = src->use_bumpmap;
    dst->bumpmap = src->bumpmap;
}

// Raygen ----------------------------------------------------------------
static __forceinline__ __device__ Vec3f reinhardToneMap(const Vec3f& color, const float white)
{
    const float l = luminance(color);
    return (color * 1.0f) / (1.0f + l / white);
}

static __forceinline__ __device__ float powerHeuristic(const float pdf1, const float pdf2)
{
    return (pdf1 * pdf1) / (pdf1 * pdf1 + pdf2 * pdf2);
}

#define NEE 0
#define MIS 1
#define MRT 0
extern "C" GLOBAL void __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int frame = params.frame;
    const Vec3ui idx(optixGetLaunchIndex());
    unsigned int seed = tea<4>(idx.x() * params.width + idx.y(), frame);

    Vec3f result(0.0f);

    int i = params.samples_per_launch;

    do
    {
        const Vec2f subpixel_jitter = UniformSampler::get2D(seed) - 0.5f;

        const Vec2f d = 2.0f * Vec2f(
            (static_cast<float>(idx.x()) + subpixel_jitter.x()) / params.width,
            (static_cast<float>(idx.y()) + subpixel_jitter.y()) / params.height
        ) - 1.0f;

        Vec3f ro, rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

        Vec3f throughput(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = Vec3f(0.0f);
        si.albedo = Vec3f(0.0f);
        si.trace_terminate = false;
        SurfaceInfo surface_info;
        surface_info.type = SurfaceType::None;
        si.surface_info = &surface_info;

        float tmax = raygen->camera.farclip / dot(rd, normalize(raygen->camera.lookat - ro));

        int depth = 0;
        for ( ;; ) {
            if ( depth >= params.max_depth )
				break;

            // Russian roulette
            if (depth >= 3) {
                float max_throughput = fmaxf(throughput.x(), fmaxf(throughput.y(), throughput.z()));
                float continue_prob = fminf(0.95f, max_throughput);

                if (rnd(si.seed) > continue_prob)
                    break;
                throughput /= continue_prob;
            }

            trace(params.handle, ro, rd, 0.01f, tmax, 0, &si);

            if (si.trace_terminate) {
                result += si.emission * throughput;
                break;
            }

            // Get emission from area emitter
            if ( si.surface_info->type == SurfaceType::AreaEmitter )
            {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    surface_info.callable_id.bsdf, &si, surface_info.data);
                result += si.emission * throughput;

                if (si.trace_terminate)
                    break;
            }
            // Specular sampling
            else if (+(si.surface_info->type & SurfaceType::Delta))
            {
                Vec3f scattered = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
                    surface_info.callable_id.sample, &si, surface_info.data);
                si.wi = scattered;

                Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                    surface_info.callable_id.bsdf, &si, surface_info.data, si.wi);
                throughput *= bsdf;
            }
            // Rough surface sampling with applying MIS
            else if ( +(si.surface_info->type & SurfaceType::Rough) )
            {
                uint32_t seed = si.seed;
                AreaEmitterInfo light;
                if (params.num_lights > 0) {
                    const int light_id = rndInt(seed, 0, params.num_lights - 1);
                    light = params.lights[light_id];
                }

                float pdf = 0.0f;
                // BSDF sampling
                Vec3f scattered = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
                    surface_info.callable_id.sample, &si, surface_info.data);
#if NEE
                float weight = 0.0f;
                si.wi = scattered;

                if (params.num_lights > 0)
                {
                    LightInteraction li;
                    // Sampling light point
                    optixDirectCall<void, const AreaEmitterInfo&, const Vec3f&, LightInteraction&, uint32_t&>(
                        light.sample_id, light, si.p, li, seed);
                    Vec3f to_light = li.p - si.p;
                    const Vec3f unit_wi = normalize(to_light);
                    const float dist_to_light = length(to_light);
                    
                    // Apply faceforward only for two-sided emitters
                    const AreaEmitter::Data* area = reinterpret_cast<AreaEmitter::Data*>(light.surface_info->data);
                    if (area->twosided)
                        li.n = faceforward(li.n, -unit_wi, li.n);
                    
                    const float cos_theta_light = dot(li.n, -unit_wi);
                    const float cos_theta_surface = dot(si.shading.n, unit_wi);

                    // For light pdf
                    if (cos_theta_light > 0.0f && cos_theta_surface > 0.0f){
                        const float t_shadow = dist_to_light - 1e-3f;
                        // Trace shadow ray
                        const bool occluded = shadowTrace(
                            params.handle, si.p, normalize(to_light), 1e-3f, t_shadow);

                        // Next Event Estimation
                        if (!occluded)
                        {
                            Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                                surface_info.callable_id.bsdf, &si, surface_info.data, unit_wi);

                            // Calculate MIS weight
                            SurfaceInteraction light_si;
                            light_si.shading.uv = li.uv;
                            light_si.shading.n = li.n;
                            light_si.wo = unit_wi;
                            light_si.surface_info = light.surface_info;

                            // Get emittance from area light
                            optixDirectCall<void, SurfaceInteraction*, void*>(
                                light_si.surface_info->callable_id.bsdf, &light_si, light_si.surface_info->data);

                            result += light_si.emission * bsdf * cos_theta_surface * throughput / li.pdf;
                        }
                    }

                    // For bsdf pdf
                    {
                        si.wi = scattered;
                        const Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                            surface_info.callable_id.bsdf, &si, surface_info.data, si.wi);

                        float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                            surface_info.callable_id.pdf, &si, surface_info.data, si.wi);

                        throughput *= bsdf / bsdf_pdf;
                    }
                }
#elif MIS      
                if (params.num_lights > 0)
                {
                    LightInteraction li;
                    // Sampling light point
                    optixDirectCall<void, const AreaEmitterInfo&, const Vec3f&, LightInteraction&, uint32_t&>(
                        light.sample_id, light, si.p, li, seed);
                    Vec3f to_light = li.p - si.p;
                    const float dist_to_light = length(to_light);

                    float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                    surface_info.callable_id.pdf, &si, surface_info.data, scattered);

                    pdf = bsdf_pdf;

                    // For light pdf
                    {
                        const float t_shadow = dist_to_light - 1e-3f;
                        // Trace shadow ray
                        const bool hit_object = shadowTrace(
                            params.handle, si.p, normalize(to_light), 1e-3f, t_shadow);

                        // Next Event Estimation
                        if (!hit_object)
                        {
                            const Vec3f unit_wi = normalize(to_light);
                            const Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                                surface_info.callable_id.bsdf, &si, surface_info.data, unit_wi);

                            float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                                surface_info.callable_id.pdf, &si, surface_info.data, unit_wi);

                            // convert unit of bsdf_pdf from [sr^-1] to [m^-2]
                            const float cos_theta = dot(-unit_wi, li.n);
                            // bsdf_pdf *= pow2(dist_to_light) / cos_theta;
                            
                            const float light_pdf = li.pdf;

                            // Calculate MIS weight
                            const float weight = powerHeuristic(light_pdf, bsdf_pdf);
                            SurfaceInteraction light_si;
                            light_si.shading.uv = li.uv;
                            light_si.shading.n = li.n;
                            light_si.wo = unit_wi;
                            light_si.surface_info = light.surface_info;

                            optixDirectCall<void, SurfaceInteraction*, void*>(
                                light_si.surface_info->callable_id.bsdf, &light_si, light_si.surface_info->data);
                            
                            result += weight * light_si.emission * bsdf * throughput / li.pdf;
                        }
                    }

                    // For bsdf pdf
                    {
                        si.wi = scattered;
                        const Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                            surface_info.callable_id.bsdf, &si, surface_info.data, si.wi);

                        float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                            surface_info.callable_id.pdf, &si, surface_info.data, si.wi);
                        const float cos_theta = dot(-si.wi, li.n);
                        const float sample_bsdf_pdf = bsdf_pdf * pow2(dist_to_light) / cos_theta;

                        const float light_pdf = optixContinuationCall<float, const AreaEmitterInfo&, const Vec3f&, const Vec3f&, LightInteraction&>(
                            light.sample_id, light, si.p, si.wi, li);

                        const float weight = powerHeuristic(bsdf_pdf, light_pdf);
                        throughput *= weight * bsdf / bsdf_pdf;
                    }
                }
#elif MRT
                si.wi = scattered;

                // Evaluate PDF depends on BSDF
                float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                    surface_info.callable_id.pdf, &si, surface_info.data, si.wi);

                // Evaluate BSDF
                Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                    surface_info.callable_id.bsdf, &si, surface_info.data, si.wi);

                throughput *= bsdf / bsdf_pdf;
#endif
                si.seed = seed;
            }

            // Make tmax large except for when the primary ray
            tmax = 1e8f;
            
            ro = si.p;
            rd = si.wi;

            ++depth;
        }
    } while (--i);

    const unsigned int image_index = idx.y() * params.width + idx.x();
    
    if (!result.isValid()) {
        result = 0.0f;
    }

    Vec3f accum_color = result / static_cast<float>(params.samples_per_launch);

    if (frame > 0)
    {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_color_prev = Vec3f(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = Vec4f(accum_color, 1.0f);
    Vec3u color = make_color(reinhardToneMap(accum_color, params.white));
    params.result_buffer[image_index] = Vec4u(color, 255);
}

// Miss ----------------------------------------------------------------
extern "C" GLOBAL void __miss__envmap()
{
    pgMissData* data = (pgMissData*)optixGetSbtDataPointer();
    auto* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    Ray ray = getWorldRay();

    Shading shading;
    float t;
    const Sphere::Data env_sphere{ Vec3f(0.0f), 1e8f };
    pgIntersectionSphere(&env_sphere, ray, &shading, &t);

    si->shading.uv = shading.uv;
    // Use sphere normal for environment map (should point outward along ray direction)
    si->shading.n = shading.n;
    si->trace_terminate = true;
    si->surface_info->type = SurfaceType::None;
    si->emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data);
}

extern "C" GLOBAL void __miss__shadow()
{
    setPayload<0>(0);
}

// Hitgroups -------------------------------------------------------------------------------
// Plane -------------------------------------------------------------------------------
static __forceinline__ __device__ bool hitPlane(
    const Plane::Data* plane, const Vec3f& o, const Vec3f& v, 
    const float tmin, const float tmax, SurfaceInteraction& si)
{
    const Vec2f min = plane->min;
    const Vec2f max = plane->max;
    
    const float t = -o.y() / v.y();
    const float x = o.x() + t * v.x();
    const float z = o.z() + t * v.z();

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && tmin < t && t < tmax)
    {
        si.shading.uv = Vec2f((x - min.x()) / (max.x() - min.x()), (z - min.y()) / (max.y() - min.y()));
        si.shading.n = Vec3f(0, 1, 0);
        si.t = t;
        si.p = o + t * v;
        return true;
    }
    return false;
}

extern "C" GLOBAL void __intersection__plane() {
    const HitgroupData* data = (HitgroupData*)optixGetSbtDataPointer();
    const Plane::Data* plane = reinterpret_cast<Plane::Data*>(data->shape_data);

    Ray ray = getLocalRay();

    Shading shading = {};
    float time = 0.0f;
    if (pgIntersectionPlane(plane, ray, &shading, &time)) {
        optixReportIntersection(time, 0, Vec3f_as_ints(shading.n), Vec2f_as_ints(shading.uv));
    }
}

extern "C" GLOBAL void __closesthit__plane()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);
    Vec2f uv = getVec2fFromAttribute<3>();

    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = uv;
    copySurfaceInfo(si->surface_info, data->surface_info);
    si->shading.dpdu = optixTransformNormalFromObjectToWorldSpace(Vec3f(1.0f, 0.0f, 0.0f));
    si->shading.dpdv = optixTransformNormalFromObjectToWorldSpace(Vec3f(0.0f, 0.0f, 1.0f));
}

extern "C" DEVICE void __direct_callable__sample_light_plane(
    const AreaEmitterInfo& area_info, const Vec3f& p, LightInteraction& li, uint32_t& seed)
{
    const auto* plane = (Plane::Data*)area_info.shape_data;

    // Sample local point on the area emitter
    const float x = rnd(seed, plane->min.x(), plane->max.x());
    const float z = rnd(seed, plane->min.y(), plane->max.y());
    Vec3f rnd_p(x, 0.0f, z);
    rnd_p = area_info.objToWorld.pointMul(rnd_p);
    li.p = rnd_p;
    li.n = normalize(area_info.objToWorld.vectorMul(Vec3f(0, 1, 0)));
    li.uv = Vec2f((x - plane->min.x()) / (plane->max.x() - plane->min.x()), (z - plane->min.y()) / (plane->max.y() - plane->min.y()));

    const Vec3f corner0 = area_info.objToWorld.pointMul(Vec3f(plane->min.x(), 0.0f, plane->min.y()));
    const Vec3f corner1 = area_info.objToWorld.pointMul(Vec3f(plane->max.x(), 0.0f, plane->min.y()));
    const Vec3f corner2 = area_info.objToWorld.pointMul(Vec3f(plane->min.x(), 0.0f, plane->max.y()));
    li.area = length(cross(corner1 - corner0, corner2 - corner0));

    const Vec3f wi = rnd_p - p;
    const float t = length(wi);
    const float cos_theta = fabs(dot(li.n, normalize(wi)));
    if (cos_theta < math::eps)
        li.pdf = 0.0f;
    li.pdf = (t * t) / (li.area * cos_theta);
}

extern "C" DEVICE float __continuation_callable__pdf_light_plane(
    const AreaEmitterInfo& area_info, const Vec3f& p, const Vec3f& wi, LightInteraction& li)
{
    const auto* plane = (Plane::Data*)area_info.shape_data;

    const Vec3f local_p = area_info.worldToObj.pointMul(p);
    const Vec3f local_wi = area_info.worldToObj.vectorMul(wi);
    SurfaceInteraction si;
    if (!hitPlane(plane, local_p, local_wi, 0.01f, 1e16f, si))
        return 0.0f;

    const float distance_squared = si.t * si.t;
    const float cosine = fabs(dot(li.n, wi));
    if (cosine < math::eps)
        return 0.0f;
    return distance_squared / (cosine * li.area);
}

// Sphere -------------------------------------------------------------------------------
static __forceinline__ __device__ bool hitSphere(
    const Sphere::Data* sphere, const Vec3f& o, const Vec3f& v, 
    const float tmin, const float tmax, SurfaceInteraction& si)
{
    const Vec3f center = sphere->center;
    const float radius = sphere->radius;

    const Vec3f oc = o - center;
    const float a = dot(v, v);
    const float half_b = dot(oc, v);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant <= 0.0f) return false;

    const float sqrtd = sqrtf(discriminant);

    float t = (-half_b - sqrtd) / a;
    if (t < tmin || tmax < t)
    {
        t = (-half_b + sqrtd) / a;
        if (t < tmin || tmax < t)
            return false;
    }

    si.t = t;
    si.p = o + t * v;
    si.shading.n = si.p / radius;
    si.shading.uv = pgGetSphereUV(si.shading.n);
    return true;
}

extern "C" GLOBAL void __intersection__sphere() {
    const HitgroupData* data = (HitgroupData*)optixGetSbtDataPointer();
    const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

    Ray ray = getLocalRay();

    Shading shading = {};
    float time = 0.0f;
    if (pgIntersectionSphere(sphere, ray, &shading, &time)) {
        optixReportIntersection(time, 0, Vec3f_as_ints(shading.n), Vec2f_as_ints(shading.uv));
    }
}

extern "C" GLOBAL void __closesthit__sphere() {
    const auto* data = (HitgroupData*)optixGetSbtDataPointer();

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);
    Vec2f uv = getVec2fFromAttribute<3>();

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = uv;
    copySurfaceInfo(si->surface_info, data->surface_info);

    float phi = atan2(local_n.z(), local_n.x());
    if (phi < 0) phi += 2.0f * math::pi;
    const float theta = acos(local_n.y());
    const Vec3f dpdu = Vec3f(-math::two_pi * local_n.z(), 0, math::two_pi * local_n.x());
    const Vec3f dpdv = math::pi * Vec3f(local_n.y() * cos(phi), -sin(theta), local_n.y() * sin(phi));
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu.toCUVec()));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv.toCUVec()));
}

extern "C" DEVICE float __continuation_callable__pdf_light_sphere(
    const AreaEmitterInfo& area_info, const Vec3f& p, const Vec3f& wi)
{
    const auto* sphere = (Sphere::Data*)area_info.shape_data;
    const Vec3f local_p = area_info.worldToObj.pointMul(p);
    const Vec3f local_wi = area_info.worldToObj.vectorMul(wi);
    
    SurfaceInteraction si;
    if (!hitSphere(sphere, local_p, local_wi, 0.01f, 1e16f, si))
        return 0.0f;

    const Vec3f center = sphere->center;
    const float radius = sphere->radius;
    const float cos_theta_max = sqrtf(1.0f - radius * radius / pow2(length(center - local_p)));
    const float solid_angle = math::two_pi * (1.0f - cos_theta_max);
    return 1.0f / solid_angle;
}

extern "C" DEVICE Vec3f __direct_callable__sample_light_sphere(
    const AreaEmitterInfo& area_info, const Vec3f& p, uint32_t& seed)
{
    const auto* sphere = (Sphere::Data*)area_info.shape_data;
    const Vec3f center = sphere->center;
    const Vec3f local_p = area_info.worldToObj.pointMul(p);
    const Vec3f oc = center - local_p;
    float distance_squared = dot(oc, oc);
    Onb onb(normalize(oc));
    Vec3f to_light = randomSampleToSphere(seed, sphere->radius, distance_squared);
    onb.inverseTransform(to_light);
    return normalize(area_info.objToWorld.vectorMul(to_light));
}

// Triangle mesh -------------------------------------------------------------------------------
extern "C" GLOBAL void __closesthit__mesh()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const TriangleMesh::Data* mesh = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Shading shading = pgGetMeshShading(mesh, optixGetTriangleBarycentrics(), optixGetPrimitiveIndex());

    Vec3f mesh_n = shading.n;
    Vec3f n(0.0f);
    if (data->surface_info->use_bumpmap) {
        n = optixDirectCall<Vec3f, Vec2f&, void*>(data->surface_info->bumpmap.prg_id, shading.uv, data->surface_info->bumpmap.data);
        Onb onb(mesh_n);
        onb.inverseTransform(n);
        shading.n = normalize(n);
    }

    // Transform shading from object to world space
    shading.n = normalize(optixTransformNormalFromObjectToWorldSpace(shading.n));
    shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdu));
    shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdv));

    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();
    si->p = ray.at(ray.tmax);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    copySurfaceInfo(si->surface_info, data->surface_info);
}

extern "C" GLOBAL void __closesthit__shadow()
{
    // Hit to surface
    setPayload<0>(1);
}

// Surfaces -------------------------------------------------------------------------------
// Diffuse -----------------------------------------------------------------------------------------------
extern "C" DEVICE Vec3f __direct_callable__sample_diffuse(SurfaceInteraction* si, void* mat_data) {
    const auto* diffuse = (Diffuse::Data*)mat_data;

    if (diffuse->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    si->trace_terminate = false;
    uint32_t seed = si->seed;
    Vec2f u = UniformSampler::get2D(seed);
    Vec3f wi = cosineSampleHemisphere(u[0], u[1]);
    Onb onb(si->shading.n);
    onb.inverseTransform(wi);
    si->seed = seed;
    return normalize(wi);
}

extern "C" DEVICE Vec3f __continuation_callable__bsdf_diffuse(SurfaceInteraction* si, void* mat_data, const Vec3f& wi)
{
    const auto* diffuse = (Diffuse::Data*)mat_data;
    const Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        diffuse->texture.prg_id, si, diffuse->texture.data);
    si->albedo = albedo;
    si->emission = Vec3f(0.0f);
    const float cosine = fmaxf(0.0f, dot(si->shading.n, wi));
    return albedo * cosine * math::inv_pi;
}

extern "C" DEVICE float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* mat_data, const Vec3f& wi)
{
    const float cosine = fmaxf(0.0f, dot(si->shading.n, wi));
    return cosine * math::inv_pi;
}

// Dielectric --------------------------------------------------------------------------------------------
extern "C" DEVICE Vec3f __direct_callable__sample_dielectric(SurfaceInteraction* si, void* mat_data) {
    const auto* dielectric = (Dielectric::Data*)mat_data;

    float ni = 1.000292f; // air
    float nt = dielectric->ior;  // ior specified 
    float cosine = dot(si->wo, si->shading.n);
    bool into = cosine < 0;
    Vec3f outward_normal = into ? si->shading.n : -si->shading.n;

    if (!into) swap(ni, nt);

    cosine = fabs(cosine);
    float sine = sqrtf(1.0f - cosine*cosine);
    bool cannot_refract = ni * sine > nt;

    float reflect_prob = fresnel(cosine, ni, nt);
    unsigned int seed = si->seed;

    Vec3f wi;
    if (cannot_refract || reflect_prob > UniformSampler::get1D(seed))
        wi = reflect(si->wo, outward_normal);
    else    
        wi = refract(si->wo, outward_normal, cosine, ni, nt);
    si->trace_terminate = false;
    si->seed = seed;
    return wi;
}

extern "C" DEVICE Vec3f __continuation_callable__bsdf_dielectric(SurfaceInteraction* si, void* mat_data, const Vec3f& /* wi */)
{
    const auto* dielectric = (Dielectric::Data*)mat_data;
    si->emission = Vec3f(0.0f);
    Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        dielectric->texture.prg_id, si, dielectric->texture.data);
    si->albedo = albedo;
    return albedo;
}

extern "C" DEVICE float __direct_callable__pdf_dielectric(SurfaceInteraction* si, void* mat_data, const Vec3f& /* wi */)
{
    return 1.0f;
}

// Conductor --------------------------------------------------------------------------------------------
extern "C" DEVICE Vec3f __direct_callable__sample_conductor(SurfaceInteraction* si, void* mat_data)
{
    const auto* conductor = (Conductor::Data*)mat_data;
    if (conductor->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    si->trace_terminate = false;
    return reflect(si->wo, si->shading.n);
}

extern "C" DEVICE Vec3f __continuation_callable__bsdf_conductor(SurfaceInteraction* si, void* mat_data, const Vec3f& /* wi */)
{
    const auto* conductor = (Conductor::Data*)mat_data;
    si->emission = Vec3f(0.0f);
    Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        conductor->texture.prg_id, si, conductor->texture.data);
    si->albedo = albedo;
    return albedo;
}

extern "C" DEVICE float __direct_callable__pdf_conductor(SurfaceInteraction* si, void* mat_data, const Vec3f& /* wi */)
{
    return 1.0f;
}

// Disney BRDF ------------------------------------------------------------------------------------------
extern "C" DEVICE Vec3f __direct_callable__sample_disney(SurfaceInteraction* si, void* mat_data)
{
    const auto* disney = (Disney::Data*)mat_data;

    if (disney->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    unsigned int seed = si->seed;
    Vec2f u = UniformSampler::get2D(seed);
    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    Onb onb(si->shading.n);

    Vec3f wi;
    if (UniformSampler::get1D(seed) < diffuse_ratio)
    {
        Vec3f w_in = cosineSampleHemisphere(u[0], u[1]);
        onb.inverseTransform(w_in);
        wi = normalize(w_in);
    }
    else
    {
        float gtr2_ratio = 1.0f / (1.0f + disney->clearcoat);
        Vec3f h;
        const float alpha = fmaxf(0.001f, disney->roughness);
        if (UniformSampler::get1D(seed) < gtr2_ratio)
            h = sampleGGX(u[0], u[1], alpha);
        else
            h = sampleGTR1(u[0], u[1], alpha);
        onb.inverseTransform(h);
        wi = normalize(reflect(si->wo, h));
    }
    si->trace_terminate = false;
    si->seed = seed;
    return wi;
}

/**
 * @ref: https://rayspace.xyz/CG/contents/Disney_principled_BRDF/
 * 
 * @note 
 * ===== Prefix =====
 * F : fresnel 
 * f : brdf function
 * G : geometry function
 * D : normal distribution function
 */
extern "C" DEVICE Vec3f __continuation_callable__bsdf_disney(SurfaceInteraction* si, void* mat_data, const Vec3f& wi)
{   
    const auto* disney = (Disney::Data*)mat_data;
    si->emission = Vec3f(0.0f);

    const Vec3f V = -normalize(si->wo);
    const Vec3f L = normalize(wi);
    const Vec3f N = normalize(si->shading.n);

    const float NdotV = fabs(dot(N, V));
    const float NdotL = fabs(dot(N, L));

    if (NdotV == 0.0f || NdotL == 0.0f)
        return Vec3f(0.0f);

    const Vec3f H = normalize(V + L);
    const float NdotH = dot(N, H);
    const float LdotH /* = VdotH */ = dot(L, H);

    const Vec3f base_color = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        disney->albedo.prg_id, si, disney->albedo.data);
    si->albedo = base_color;

    // Diffuse term (diffuse, subsurface, sheen) ======================
    // Diffuse
    const float Fd90 = 0.5f + 2.0f * disney->roughness * LdotH*LdotH;
    const float FVd90 = fresnelSchlickT(NdotV, Fd90);
    const float FLd90 = fresnelSchlickT(NdotL, Fd90);
    const Vec3f f_diffuse = (base_color * math::inv_pi) * FVd90 * FLd90;

    // Subsurface
    const float Fss90 = disney->roughness * LdotH*LdotH;
    const float FVss90 = fresnelSchlickT(NdotV, Fss90);
    const float FLss90 = fresnelSchlickT(NdotL, Fss90); 
    const Vec3f f_subsurface = (base_color * math::inv_pi) * 1.25f * (FVss90 * FLss90 * ((1.0f / (NdotV * NdotL)) - 0.5f) + 0.5f);

    // Sheen
    const Vec3f rho_tint = base_color / luminance(base_color);
    const Vec3f rho_sheen = lerp(Vec3f(1.0f), rho_tint, disney->sheen_tint);
    const Vec3f f_sheen = disney->sheen * rho_sheen * powf(1.0f - LdotH, 5.0f);

    // Specular term (specular, clearcoat) ============================
    // Spcular
    const Vec3f X = si->shading.dpdu;
    const Vec3f Y = si->shading.dpdv;
    const float alpha = fmaxf(0.001f, disney->roughness);
    const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
    const float ax = fmaxf(0.001f, pow2(alpha) / aspect);
    const float ay = fmaxf(0.001f, pow2(alpha) * aspect);
    const Vec3f rho_specular = lerp(Vec3f(1.0f), rho_tint, disney->specular_tint);
    const Vec3f Fs0 = lerp(0.08f * disney->specular * rho_specular, base_color, disney->metallic);
    const Vec3f FHs0 = fresnelSchlickR(LdotH, Fs0);
    const float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);
    const Vec3f f_specular = FHs0 * Ds * Gs;

    // Clearcoat
    const float Fcc = fresnelSchlickR(LdotH, 0.04f);
    const float alpha_cc = 0.1f + (0.001f - 0.1f) * disney->clearcoat_gloss; // lerp
    const float Dcc = GTR1(NdotH, alpha_cc);
    const float Gcc = smithG_GGX(NdotV, 0.25f);
    const Vec3f f_clearcoat = Vec3f( 0.25f * disney->clearcoat * (Fcc * Dcc * Gcc) );

    const Vec3f out = ( 1.0f - disney->metallic ) * ( lerp( f_diffuse, f_subsurface, disney->subsurface ) + f_sheen ) + f_specular + f_clearcoat;
    return out * clamp(NdotL, 0.0f, 1.0f);
}

/**
 * @ref http://simon-kallweit.me/rendercompo2015/report/#adaptivesampling
 * 
 * @todo Investigate correct evaluation of PDF.
 */
extern "C" DEVICE float __direct_callable__pdf_disney(SurfaceInteraction* si, void* mat_data, const Vec3f& wi)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);

    const Vec3f V = -si->wo;
    const Vec3f L = wi;
    const Vec3f N = si->shading.n;

    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    const float specular_ratio = 1.0f - diffuse_ratio;

    const float NdotL = abs(dot(N, L));
    const float NdotV = abs(dot(N, V));

    const float alpha = fmaxf(0.001f, disney->roughness);
    const float alpha_cc = lerp(0.001f, 0.1f, disney->clearcoat_gloss);
    const Vec3f H = normalize(V + L);
    const float NdotH = abs(dot(H, N));

    const float pdf_Ds = GTR2(NdotH, alpha);
    const float pdf_Dcc = GTR1(NdotH, alpha_cc);
    const float ratio = 1.0f / (1.0f + disney->clearcoat);
    const float pdf_specular = (pdf_Dcc + ratio * (pdf_Ds - pdf_Dcc));
    const float pdf_diffuse = NdotL * math::inv_pi;

    return diffuse_ratio * pdf_diffuse + specular_ratio * pdf_specular;
}

// Area emitter ------------------------------------------------------------------------------------------
extern "C" DEVICE void __direct_callable__area_emitter(SurfaceInteraction* si, void* surface_data)
{
    const auto* area = (AreaEmitter::Data*)surface_data;
    si->trace_terminate = true;
    float is_emitted = 1.0f;
    if (!area->twosided)
        is_emitted *= (float)(dot(si->wo, si->shading.n) < 0.0f);

    const Vec3f base = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        area->texture.prg_id, si, area->texture.data);
    si->albedo = base;
    si->emission = base * area->intensity * is_emitted;
}

// Textures ------------------------------------------------------------------------------------------
extern "C" DEVICE Vec3f __direct_callable__bitmap(SurfaceInteraction* si, void* tex_data) {
    const auto* image = (BitmapTexture::Data*)tex_data;
    float4 c = tex2D<float4>(image->texture, si->shading.uv.x(), si->shading.uv.y());
    return Vec3f(c);
}

extern "C" DEVICE Vec3f __direct_callable__constant(SurfaceInteraction* si, void* tex_data) {
    const auto* constant = (ConstantTexture::Data*)tex_data;
    return constant->color;
}

extern "C" DEVICE Vec3f __direct_callable__checker(SurfaceInteraction* si, void* tex_data) {
    const auto* checker = (CheckerTexture::Data*)tex_data;
    const bool is_odd = sinf(si->shading.uv.x() * math::pi * checker->scale) * sinf(si->shading.uv.y() * math::pi * checker->scale) < 0;
    return lerp(checker->color1, checker->color2, (float)is_odd);
}