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

static INLINE DEVICE void trace(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd,
    float tmin, float tmax, SurfaceInteraction* si) 
{
    uint32_t u0, u1;
    packPointer( si, u0, u1 );
    optixTrace(handle, ro.toCUVec(), rd.toCUVec(),
        tmin, tmax, 0.0f,
        OptixVisibilityMask( 1 ), OPTIX_RAY_FLAG_NONE,
        (uint32_t)RayType::RADIANCE, (uint32_t)RayType::N_RAY, (uint32_t)RayType::RADIANCE,
        u0, u1 );
}

static INLINE DEVICE bool shadowTrace(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd,
    float tmin, float tmax)
{
    uint32_t hit = 0u;
    optixTrace(handle, ro.toCUVec(), rd.toCUVec(),
        tmin, tmax, 0.0f,
        OptixVisibilityMask(1), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        (uint32_t)RayType::SHADOW, (uint32_t)RayType::N_RAY, (uint32_t)RayType::SHADOW,
        hit);
    return static_cast<bool>(hit);
}

// Raygen ----------------------------------------------------------------
static __forceinline__ __device__ void getCameraRay(
    const Camera::Data& camera, const float x, const float y, Vec3f& ro, Vec3f& rd)
{
    rd = normalize(x * camera.U + y * camera.V + camera.W);
    ro = camera.origin;
}

static __forceinline__ __device__ Vec3f reinhardToneMap(const Vec3f& color, const float white)
{
    const float l = luminance(color);
    return (color * 1.0f) / (1.0f + l / white);
}

static __forceinline__ __device__ float powerHeuristic(const float pdf1, const float pdf2)
{
    return (pdf1 * pdf1) / (pdf1 * pdf1 + pdf2 * pdf2);
}

#define MIS1 0
#define MIS2 0
#define MIS3 1
#define PT 0
extern "C" __device__ void __raygen__pinhole()
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
        si.radiance_evaled = false;

        float tmax = raygen->camera.farclip / dot(rd, normalize(raygen->camera.lookat - ro));

        int depth = 0;
        for ( ;; ) {

            if ( depth >= params.max_depth )
				break;

            trace(params.handle, ro, rd, 0.01f, tmax, &si);

            if (si.trace_terminate) {
                result += si.emission * throughput;
                break;
            }

            // Get emission from area emitter
            if ( si.surface_info.type == SurfaceType::AreaEmitter )
            {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id, &si, si.surface_info.data);
                result += si.emission * throughput;
                if (si.trace_terminate)
                    break;
            }
            // Specular sampling
            else if (+(si.surface_info.type & SurfaceType::Delta))
            {
                Vec3f scattered = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id, &si, si.surface_info.data);
                si.wi = scattered;

                Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                    si.surface_info.bsdf_id, &si, si.surface_info.data, Vec3f{});
                throughput *= bsdf;
            }
            // Rough surface sampling with applying MIS
            else if ( +(si.surface_info.type & (SurfaceType::Rough | SurfaceType::Diffuse)) )
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
                    si.surface_info.sample_id, &si, si.surface_info.data);
#if MIS1
                float weight = 0.0f;

                float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                    si.surface_info.pdf_id, &si, si.surface_info.data, scattered);

                si.wi = scattered;
                pdf = bsdf_pdf;

                if (params.num_lights > 0)
                {
                    // Light sampling
                    /// @note to_light is not normalized
                    LightInteraction li;
                    optixDirectCall<void, const AreaEmitterInfo&, const Vec3f&, LightInteraction&, uint32_t&>(
                        light.sample_id, light, si.p, li, seed);

                    Vec3f to_light = li.p - si.p;

                    li.pdf *= (float)(dot(si.shading.n, to_light) <= 0);

                    const float cos_theta = dot(-scattered, li.n);

                    // Conversion from [sr^-1] to [m^-2] unit
                    float sample_bsdf_pdf = bsdf_pdf * lengthSquared(to_light) / cos_theta;
                    sample_bsdf_pdf *= (float)(cos_theta < 0.0f);
                    if (li.pdf > 0.0f && sample_bsdf_pdf > 0.0f)
                    {
                        const float bsdf_weight = powerHeuristic(sample_bsdf_pdf, li.pdf);
                        if (UniformSampler::get1D(seed) > bsdf_weight)
                            si.wi = normalize(to_light);
                        pdf = bsdf_weight * bsdf_pdf + (1.0f - bsdf_weight) * li.pdf;
                    }
                }
#elif MIS2
                const float weight = 1.0f / (params.num_lights + 1);
                if (params.num_lights > 0) {
                    // Light sampling
                    LightInteraction li;
                    optixDirectCall<void, const AreaEmitterInfo&, const Vec3f&, LightInteraction&, uint32_t&>(
                        light.sample_id, light, si.p, li, seed);

                    if (UniformSampler::get1D(seed) < weight * params.num_lights)
                        si.wi = normalize(li.p - si.p);
                    else
                        si.wi = scattered;

                    const float light_pdf = optixContinuationCall<float, const AreaEmitterInfo&, const Vec3f&, const Vec3f&, LightInteraction&>(
                        light.sample_id, light, si.p, si.wi, li);

                    pdf += (weight * params.num_lights) * light_pdf;
                }

                // Evaluate PDF depends on BSDF
                float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                    si.surface_info.pdf_id, &si, si.surface_info.data, si.wi);

                pdf += weight * bsdf_pdf;
#elif MIS3      
                if (params.num_lights > 0)
                {
                    LightInteraction li;
                    // Sampling light point
                    optixDirectCall<void, const AreaEmitterInfo&, const Vec3f&, LightInteraction&, uint32_t&>(
                        light.sample_id, light, si.p, li, seed);
                    Vec3f to_light = li.p - si.p;
                    const float dist_to_light = length(to_light);

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
                                si.surface_info.bsdf_id, &si, si.surface_info.data, unit_wi);

                            float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                                si.surface_info.pdf_id, &si, si.surface_info.data, unit_wi);

                            // convert unit of bsdf_pdf from [sr^-1] to [m^-2]
                            const float cos_theta = dot(-unit_wi, li.n);
                            bsdf_pdf *= pow2(dist_to_light) / cos_theta;
                            
                            const float light_pdf = li.pdf;

                            // Calculate MIS weight
                            const float weight = powerHeuristic(light_pdf, bsdf_pdf);
                            SurfaceInteraction light_si;
                            light_si.uv = li.uv;
                            light_si.shading.n = li.n;
                            light_si.wo = unit_wi;
                            light_si.surface_info = light.surface_info;

                            optixDirectCall<void, SurfaceInteraction*, void*>(
                                light_si.surface_info.bsdf_id, &light_si, light_si.surface_info.data);
                            
                            result += weight * light_si.emission * bsdf * throughput / li.pdf;
                        }
                    }

                    // For bsdf pdf
                    {
                        si.wi = scattered;
                        const Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                            si.surface_info.bsdf_id, &si, si.surface_info.data, si.wi);
                        
                        float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                            si.surface_info.pdf_id, &si, si.surface_info.data, si.wi);
                        const float cos_theta = dot(-si.wi, li.n);
                        const float sample_bsdf_pdf = bsdf_pdf * pow2(dist_to_light) / cos_theta;

                        const float light_pdf = optixContinuationCall<float, const AreaEmitterInfo&, const Vec3f&, const Vec3f&, LightInteraction&>(
                            light.sample_id, light, si.p, si.wi, li);

                        const float weight = powerHeuristic(sample_bsdf_pdf, light_pdf);
                        throughput *= weight * bsdf / bsdf_pdf;
                    }
                }
#elif PT
                si.wi = scattered;

                // Evaluate PDF depends on BSDF
                float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
                    si.surface_info.pdf_id, &si, si.surface_info.data, si.wi);
                pdf = bsdf_pdf;
#endif

#if !MIS3
                // Evaluate BSDF
                Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                    si.surface_info.bsdf_id, &si, si.surface_info.data, si.wi);

                throughput *= bsdf / pdf;
#endif
                si.seed = seed;
            }

            // Make tmax large except for when the primary ray
            tmax = 1e16f;
            
            ro = si.p;
            rd = si.wi;

            ++depth;
        }
    } while (--i);

    const unsigned int image_index = idx.y() * params.width + idx.x();

    if (result.x() != result.x()) result.x() = 0.0f;
    if (result.y() != result.y()) result.y() = 0.0f;
    if (result.z() != result.z()) result.z() = 0.0f;

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
extern "C" __device__ void __miss__envmap()
{
    const auto* data = (MissData*)optixGetSbtDataPointer();
    const auto* env = (EnvironmentEmitter::Data*)data->env_data;
    auto* si = getSurfaceInteraction();

    Ray ray = getWorldRay();

    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f*1e8f;
    const float discriminant = half_b * half_b - a*c;

    float sqrtd = sqrtf(discriminant);
    float t = (-half_b + sqrtd) / a;

    Vec3f p = normalize(ray.at(t));

    const float phi = atan2(p.z(), p.x());
    const float theta = asin(p.y());
    const float u = 1.0f - (phi + math::pi) / (math::two_pi);
    const float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    si->uv = Vec2f(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data);
}

extern "C" __device__ void __miss__shadow()
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
        si.uv = Vec2f((x - min.x()) / (max.x() - min.x()), (z - min.y()) / (max.y() - min.y()));
        si.shading.n = Vec3f(0, 1, 0);
        si.t = t;
        si.p = o + t * v;
        return true;
    }
    return false;
}

extern "C" __device__ void __intersection__plane()
{
    const auto* data = (HitgroupData*)optixGetSbtDataPointer();
    const auto* plane = (Plane::Data*)data->shape_data;

    const Vec2f min = plane->min;
    const Vec2f max = plane->max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y() / ray.d.y();

    const float x = ray.o.x() + t * ray.d.x();
    const float z = ray.o.z() + t * ray.d.z();

    Vec2f uv((x - min.x()) / (max.x() - min.x()), (z - min.y()) / (max.y() - min.y()));

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, Vec2f_as_ints(uv));
}

extern "C" __device__ void __closesthit__plane()
{
    const auto* data = (HitgroupData*)optixGetSbtDataPointer();

    Ray ray = getWorldRay();

    Vec3f local_n(0, 1, 0);
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n.toCUVec());
    world_n = normalize(world_n);
    Vec2f uv = getVec2fFromAttribute<0>();

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;
    si->shading.dpdu = optixTransformNormalFromObjectToWorldSpace({1.0f, 0.0f, 0.0f});
    si->shading.dpdv = optixTransformNormalFromObjectToWorldSpace({0.0f, 0.0f, 1.0f});
}

extern "C" __device__ void __direct_callable__sample_light_plane(
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

extern "C" __device__ float __continuation_callable__pdf_light_plane(
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
static __forceinline__ __device__ Vec2f getSphereUV(const Vec3f& p) 
{
    const float phi = atan2(p.z(), p.x());
    const float theta = asin(p.y());
    const float u = 1.0f - (phi + math::pi) / (math::two_pi);
    const float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    return Vec2f(u, v);
}

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
    si.uv = getSphereUV(si.shading.n);
    return true;
}

extern "C" __device__ void __intersection__sphere() 
{
    const auto* data = (HitgroupData*)optixGetSbtDataPointer();
    const auto* sphere = (Sphere::Data*)data->shape_data;

    const Vec3f center = sphere->center;
    const float radius = sphere->radius;

    Ray ray = getLocalRay();

    const Vec3f oc = ray.o - center;
    const float a = dot(ray.d, ray.d);
    const float half_b = dot(oc, ray.d);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant > 0.0f) {
        float sqrtd = sqrtf(discriminant);
        float t1 = (-half_b - sqrtd) / a;
        bool check_second = true;
        if (t1 > ray.tmin && t1 < ray.tmax) {
            Vec3f normal = normalize((ray.at(t1) - center) / radius);
            check_second = false;
            optixReportIntersection(t1, 0, Vec3f_as_ints(normal));
        }

        if (check_second) {
            float t2 = (-half_b + sqrtd) / a;
            if (t2 > ray.tmin && t2 < ray.tmax) {
                Vec3f normal = normalize((ray.at(t2) - center) / radius);
                optixReportIntersection(t2, 0, Vec3f_as_ints(normal));
            }
        }
    }
}

extern "C" __device__ void __closesthit__sphere() {
    const auto* data = (HitgroupData*)optixGetSbtDataPointer();

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n.toCUVec());
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->uv = getSphereUV(local_n);
    si->surface_info = data->surface_info;

    float phi = atan2(local_n.z(), local_n.x());
    if (phi < 0) phi += 2.0f * math::pi;
    const float theta = acos(local_n.y());
    const Vec3f dpdu = Vec3f(-math::two_pi * local_n.z(), 0, math::two_pi * local_n.x());
    const Vec3f dpdv = math::pi * Vec3f(local_n.y() * cos(phi), -sin(theta), local_n.y() * sin(phi));
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu.toCUVec()));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv.toCUVec()));
}

extern "C" __device__ float __continuation_callable__pdf_light_sphere(
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

extern "C" __device__ Vec3f __direct_callable__sample_light_sphere(
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
extern "C" __device__ void __closesthit__mesh()
{
    const auto* data = (HitgroupData*)optixGetSbtDataPointer();
    const auto* mesh = (TriangleMesh::Data*)data->shape_data;

    Ray ray = getWorldRay();
    
    const int prim_id = optixGetPrimitiveIndex();
    const Face face = mesh->faces[prim_id];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const Vec3f p0 = mesh->vertices[face.vertex_id.x()];
    const Vec3f p1 = mesh->vertices[face.vertex_id.y()];
    const Vec3f p2 = mesh->vertices[face.vertex_id.z()];

    const Vec2f texcoord0 = mesh->texcoords[face.texcoord_id.x()];
    const Vec2f texcoord1 = mesh->texcoords[face.texcoord_id.y()];
    const Vec2f texcoord2 = mesh->texcoords[face.texcoord_id.z()];
    const Vec2f texcoords = (1-u-v)*texcoord0 + u*texcoord1 + v*texcoord2;

    Vec3f n0 = mesh->normals[face.normal_id.x()];
	Vec3f n1 = mesh->normals[face.normal_id.y()];
	Vec3f n2 = mesh->normals[face.normal_id.z()];

    // Linear interpolation of normal by barycentric coordinates.
    Vec3f local_n = (1.0f-u-v)*n0 + u*n1 + v*n2;
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n.toCUVec());
    world_n = normalize(world_n);

    auto* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->uv = texcoords;
    si->surface_info = data->surface_info;

    Vec3f dpdu, dpdv;
    const Vec2f duv02 = texcoord0 - texcoord2, duv12 = texcoord1 - texcoord2;
    const Vec3f dp02 = p0 - p2, dp12 = p1 - p2;
    const float D = duv02.x() * duv12.y() - duv02.y() * duv12.x();
    bool degenerateUV = abs(D) < 1e-8f;
    if (!degenerateUV)
    {
        const float invD = 1.0f / D;
        dpdu = (duv12.y() * dp02 - duv02.y() * dp12) * invD;
        dpdv = (-duv12.x() * dp02 + duv02.x() * dp12) * invD;
    }
    if (degenerateUV || length(cross(dpdu, dpdv)) == 0.0f)
    {
        const Vec3f n = normalize(cross(p2 - p0, p1 - p0));
        Onb onb(n);
        dpdu = onb.tangent;
        dpdv = onb.bitangent;
    }
    si->shading.dpdu = normalize(optixTransformNormalFromObjectToWorldSpace(dpdu.toCUVec()));
    si->shading.dpdv = normalize(optixTransformNormalFromObjectToWorldSpace(dpdv.toCUVec()));
}

extern "C" __device__ void __closesthit__shadow()
{
    // Hit to surface
    setPayload<0>(1);
}

// Surfaces -------------------------------------------------------------------------------
// Diffuse -----------------------------------------------------------------------------------------------
extern "C" __device__ Vec3f __direct_callable__sample_diffuse(SurfaceInteraction* si, void* mat_data) {
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

extern "C" __device__ Vec3f __continuation_callable__bsdf_diffuse(SurfaceInteraction* si, void* mat_data, const Vec3f& wi)
{
    const auto* diffuse = (Diffuse::Data*)mat_data;
    const Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        diffuse->texture.prg_id, si, diffuse->texture.data);
    si->albedo = albedo;
    si->emission = Vec3f(0.0f);
    const float cosine = fmaxf(math::eps, dot(si->shading.n, wi));
    return albedo * cosine * math::inv_pi;
}

extern "C" __device__ float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* mat_data, const Vec3f& wi)
{
    const float cosine = fmaxf(math::eps, dot(si->shading.n, wi));
    return cosine * math::inv_pi;
}

// Dielectric --------------------------------------------------------------------------------------------
extern "C" __device__ Vec3f __direct_callable__sample_dielectric(SurfaceInteraction* si, void* mat_data) {
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
    si->radiance_evaled = false;
    si->trace_terminate = false;
    si->seed = seed;
    return wi;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_dielectric(SurfaceInteraction* si, void* mat_data, const Vec3f& /* wi */)
{
    const auto* dielectric = (Dielectric::Data*)mat_data;
    si->emission = Vec3f(0.0f);
    Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        dielectric->texture.prg_id, si, dielectric->texture.data);
    si->albedo = albedo;
    return albedo;
}

extern "C" __device__ float __direct_callable__pdf_dielectric(SurfaceInteraction* si, void* mat_data, const Vec3f& /* wi */)
{
    return 1.0f;
}

// Conductor --------------------------------------------------------------------------------------------
extern "C" __device__ Vec3f __direct_callable__sample_conductor(SurfaceInteraction* si, void* mat_data) 
{
    const auto* conductor = (Conductor::Data*)mat_data;
    if (conductor->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    si->trace_terminate = false;
    si->radiance_evaled = false;
    return reflect(si->wo, si->shading.n);
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_conductor(SurfaceInteraction* si, void* mat_data, const Vec3f& /* wi */)
{
    const auto* conductor = (Conductor::Data*)mat_data;
    si->emission = Vec3f(0.0f);
    Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        conductor->texture.prg_id, si, conductor->texture.data);
    si->albedo = albedo;
    return albedo;
}

extern "C" __device__ float __direct_callable__pdf_conductor(SurfaceInteraction* si, void* mat_data, const Vec3f& /* wi */)
{
    return 1.0f;
}

// Disney BRDF ------------------------------------------------------------------------------------------
extern "C" __device__ Vec3f __direct_callable__sample_disney(SurfaceInteraction* si, void* mat_data)
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
    si->radiance_evaled = false;
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
extern "C" __device__ Vec3f __continuation_callable__bsdf_disney(SurfaceInteraction* si, void* mat_data, const Vec3f& wi)
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
        disney->base.prg_id, si, disney->base.data);
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
extern "C" __device__ float __direct_callable__pdf_disney(SurfaceInteraction* si, void* mat_data, const Vec3f& wi)
{
    const DisneyData* disney = reinterpret_cast<DisneyData*>(mat_data);

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
extern "C" __device__ void __direct_callable__area_emitter(SurfaceInteraction* si, void* surface_data)
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
extern "C" __device__ Vec3f __direct_callable__bitmap(SurfaceInteraction* si, void* tex_data) {
    const auto* image = (BitmapTexture::Data*)tex_data;
    float4 c = tex2D<float4>(image->texture, si->uv.x(), si->uv.y());
    return Vec3f(c);
}

extern "C" __device__ Vec3f __direct_callable__constant(SurfaceInteraction* si, void* tex_data) {
    const auto* constant = (ConstantTexture::Data*)tex_data;
    return constant->color;
}

extern "C" __device__ Vec3f __direct_callable__checker(SurfaceInteraction* si, void* tex_data) {
    const auto* checker = (CheckerTexture::Data*)tex_data;
    const bool is_odd = sinf(si->uv.x() * math::pi * checker->scale) * sinf(si->uv.y() * math::pi * checker->scale) < 0;
    return lerp(checker->color1, checker->color2, (float)is_odd);
}