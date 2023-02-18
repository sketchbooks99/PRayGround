#include <prayground/prayground.h>

#include "params.h"

using namespace prayground;

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

INLINE DEVICE void trace(
    OptixTraversableHandle handle,
    const Vec3f& ro, const Vec3f& rd,
    float tmin, float tmax,
    uint32_t ray_type,
    uint32_t ray_count,
    SurfaceInteraction* si
)
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle,
        ro, rd,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        ray_type,
        1,
        ray_type,
        u0, u1
    );
}

// --------------------------------------------------------------------------
// Raygen 
// --------------------------------------------------------------------------
static __forceinline__ __device__ Vec3f reinhardToneMap(const Vec3f& color, const float white)
{
    const float l = luminance(color);
    return (color * 1.0f) / (1.0f + l / white);
}

extern "C" __device__ void __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int frame = params.frame;
    const Vec3ui idx(optixGetLaunchIndex());
    unsigned int seed = tea<4>(idx.y() * params.width + idx.x(), frame);

    Vec3f result(0.0f);
    Vec3f normal(0.0f);
    Vec3f albedo(0.0f);

    int spl = params.samples_per_launch;

    for (int i = 0; i < spl; i++)
    {
        const Vec2f jitter = UniformSampler::get2D(seed) - 0.5f;
        const Vec2f res(params.width, params.height);
        const Vec2f d = 2.0f * ((Vec2f(idx.x(), idx.y()) + jitter) / res) - 1.0f;

        Vec3f ro, rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

        Vec3f throughput(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = Vec3f(0.0f);
        si.albedo = Vec3f(0.0f);
        si.shading.n = Vec3f(0.0f);
        si.trace_terminate = false;

        float tmax = raygen->camera.farclip / dot(rd, normalize(raygen->camera.W));

        int depth = 0;

        while (true)
        {
            if (depth >= params.max_depth)
                break;

            trace(params.handle, ro, rd, 0.01f, tmax, /* ray_type = */ 0, 1, &si);

            if (si.trace_terminate)
            {
                result += throughput * si.emission;
                break;
            }

            // Get emission from area emitter
            if (si.surface_info.type == SurfaceType::AreaEmitter)
            {
                // Evaluate emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data);
                result += throughput * si.emission;
                
                if (depth == 0)
                {
                    albedo += si.albedo;
                    normal += si.shading.n;
                }

                if (si.trace_terminate)
                    break;
            }
            // Specular sampling
            else if (+(si.surface_info.type & SurfaceType::Delta))
            {
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.sample, &si, si.surface_info.data);

                // Evaluate BSDF
                const Vec3f bsdf_val = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data);
                throughput *= bsdf_val;
            }
            // Rough surface sampling
            else if (+(si.surface_info.type & (SurfaceType::Rough | SurfaceType::Diffuse)))
            {
                // Importance sampling according to the BSDF
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.sample, &si, si.surface_info.data);

                // Evaluate PDF depends on BSDF
                float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.pdf, &si, si.surface_info.data);

                // Evaluate BSDF
                Vec3f bsdf_val = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data);

                if (bsdf_pdf == 0.0f) break;

                throughput *= bsdf_val / bsdf_pdf;
            }

            if (depth == 0)
            {
                albedo += si.albedo;
                normal += si.shading.n;
            }

            tmax = 1e16f;

            ro = si.p;
            rd = si.wi;

            depth++;
        }
    }

    const uint32_t image_idx = idx.y() * params.width + idx.x();

    if (result.x() != result.x()) result.x() = 0.0f;
    if (result.y() != result.y()) result.y() = 0.0f;
    if (result.z() != result.z()) result.z() = 0.0f;

    Vec3f accum = result / (float)spl;
    albedo = albedo / (float)spl;
    normal = normal / (float)spl;

    if (frame > 0)
    {
        const float a = 1.0f / (float)(frame + 1);
        const Vec3f accum_prev(params.accum_buffer[image_idx]);
        accum = lerp(accum_prev, accum, a);
        const Vec3f albedo_prev(params.albedo_buffer[image_idx]);
        const Vec3f normal_prev(params.normal_buffer[image_idx]);
        albedo = lerp(albedo_prev, albedo / (float)spl, a);
        normal = lerp(normal_prev, normal / (float)spl, a);
    }
    params.accum_buffer[image_idx] = Vec4f(accum, 1.0f);
    Vec3u color = make_color(accum);
    params.result_buffer[image_idx] = Vec4f(color2float(color), 1.0f);
    params.normal_buffer[image_idx] = Vec4f(normal, 1.0f);
    params.albedo_buffer[image_idx] = Vec4f(albedo, 1.0f);
}

// --------------------------------------------------------------------------
// Miss
// --------------------------------------------------------------------------
extern "C" __device__ void __miss__envmap()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    EnvironmentEmitter::Data* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
    SurfaceInteraction* si = getSurfaceInteraction();

    Ray ray = getWorldRay();

    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f * 1e8f;
    const float discriminant = half_b * half_b - a * c;

    float sqrtd = sqrtf(discriminant);
    float t = (-half_b + sqrtd) / a;

    Vec3f p = normalize(ray.at(t));

    float phi = atan2(p.z(), p.x());
    float theta = asin(p.y());
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) * math::inv_pi;
    si->shading.uv = Vec2f(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Vec4f, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data);
}

// --------------------------------------------------------------------------
// Callables for surface
// --------------------------------------------------------------------------
// Diffuse -----------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_diffuse(SurfaceInteraction* si, void* mat_data) {
    const auto* diffuse = reinterpret_cast<Diffuse::Data*>(mat_data);
    si->wi = importanceSamplingDiffuse(diffuse, si->wo, si->shading, si->seed);
    si->trace_terminate = false;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(mat_data);
    const Vec4f albedo = optixDirectCall<Vec4f, const Vec2f&, void*>(diffuse->texture.prg_id, si->shading.uv, diffuse->texture.data);
    si->albedo = albedo;
    si->emission = Vec3f(0.0f);
    return albedo * getDiffuseBRDF(si->wi, si->shading.n);
}

extern "C" __device__ float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
    return getDiffusePDF(si->wi, si->shading.n);
}

// Dielectric --------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_dielectric(SurfaceInteraction* si, void* mat_data) {
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(mat_data);
    si->wi = samplingSmoothDielectric(dielectric, si->wo, si->shading, si->seed);
    si->trace_terminate = false;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_dielectric(SurfaceInteraction* si, void* mat_data)
{
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(mat_data);
    const Vec4f albedo = optixDirectCall<Vec4f, const Vec2f&, void*>(dielectric->texture.prg_id, si->shading.uv, dielectric->texture.data);
    si->albedo = Vec3f(1.0f);
    si->emission = Vec3f(0.0f);
    return albedo;
}

extern "C" __device__ float __direct_callable__pdf_dielectric(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}

// Conductor --------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_conductor(SurfaceInteraction* si, void* mat_data) {
    const auto* conductor = reinterpret_cast<Conductor::Data*>(mat_data);
    if (conductor->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    si->wi = reflect(si->wo, si->shading.n);
    si->trace_terminate = false;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_conductor(SurfaceInteraction* si, void* mat_data)
{
    const auto* conductor = reinterpret_cast<Conductor::Data*>(mat_data);
    si->emission = Vec3f(0.0f);
    Vec4f albedo = optixDirectCall<Vec4f, SurfaceInteraction*, void*>(conductor->texture.prg_id, si, conductor->texture.data);
    si->albedo = Vec3f(albedo);
    return si->albedo;
}

extern "C" __device__ float __direct_callable__pdf_conductor(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}

// Disney BRDF ------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_disney(SurfaceInteraction* si, void* mat_data)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);
    si->wi = importanceSamplingDisney(disney, si->wo, si->shading, si->seed);
    si->trace_terminate = false;
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
extern "C" __device__ Vec3f __continuation_callable__bsdf_disney(SurfaceInteraction* si, void* mat_data)
{   
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);
    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(disney->base.prg_id, si->shading.uv, disney->base.data);
    return getDisneyBRDF(disney, si->wo, si->wi, si->shading, base);
}

/**
 * @ref http://simon-kallweit.me/rendercompo2015/report/#adaptivesampling
 * 
 * @todo Investigate correct evaluation of PDF.
 */
extern "C" __device__ float __direct_callable__pdf_disney(SurfaceInteraction* si, void* mat_data)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);
    return getDisneyPDF(disney, si->wo, si->wi, si->shading);
}

// Area emitter ------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__area_emitter(SurfaceInteraction* si, void* surface_data)
{
    const AreaEmitter::Data* area = reinterpret_cast<AreaEmitter::Data*>(surface_data);
    si->trace_terminate = true;
    float is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    if (area->twosided)
    {
        is_emitted = 1.0f;
        si->shading.n = faceforward(si->shading.n, -si->wi, si->shading.n);
    }

    // Get texture
    auto base = optixDirectCall<Vec4f, SurfaceInteraction*, void*>(
        area->texture.prg_id, si, area->texture.data);
    si->albedo = Vec3f(base);
    
    si->emission = si->albedo * area->intensity * is_emitted;
}

// --------------------------------------------------------------------------
// Callables for texture
// --------------------------------------------------------------------------
extern "C" __device__ Vec4f __direct_callable__bitmap(SurfaceInteraction* si, void* tex_data) {
    const auto* image = reinterpret_cast<BitmapTexture::Data*>(tex_data);
    float4 c = tex2D<float4>(image->texture, si->shading.uv.x(), si->shading.uv.y());
    return c;
}

extern "C" __device__ Vec4f __direct_callable__constant(SurfaceInteraction* si, void* tex_data) {
    const auto* constant = reinterpret_cast<ConstantTexture::Data*>(tex_data);
    return constant->color;
}

extern "C" __device__ Vec4f __direct_callable__checker(SurfaceInteraction* si, void* tex_data) {
    const auto* checker = reinterpret_cast<CheckerTexture::Data*>(tex_data);
    const bool is_odd = sinf(si->shading.uv.x() * math::pi * checker->scale) * sinf(si->shading.uv.y() * math::pi * checker->scale) < 0;
    return lerp(checker->color1, checker->color2, static_cast<float>(is_odd));
}

// --------------------------------------------------------------------------
// Hitgroups
// --------------------------------------------------------------------------
// Plane -------------------------------------------------------------------------------
static __forceinline__ __device__ bool hitPlane(
    const Plane::Data* plane, const Vec3f& o, const Vec3f& v, const float tmin, const float tmax, SurfaceInteraction& si)
{
    const Vec2f min = plane->min;
    const Vec2f max = plane->max;
    
    const float t = -o.y() / v.y();
    const float x = o.x() + t * v.x();
    const float z = o.z() + t * v.z();

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && tmin < t && t < tmax)
    {
        si.shading.uv = Vec2f((x - min.x()) / (max.x() - min.x()), (z - min.y()) / max.y() - min.y());
        si.shading.n = Vec3f(0, 1, 0);
        si.t = t;
        si.p = o + t*v;
        return true;
    }
    return false;
}

extern "C" __device__ void __intersection__plane()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* plane = reinterpret_cast<Plane::Data*>(data->shape_data);

    const Vec2f min = plane->min;
    const Vec2f max = plane->max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y() / ray.d.y();

    const float x = ray.o.x() + t * ray.d.x();
    const float z = ray.o.z() + t * ray.d.z();

    Vec2f uv = Vec2f((x - min.x()) / (max.x() - min.x()), (z - min.y()) / (max.y() - min.y()));

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, Vec2f_as_ints(uv));
}

extern "C" __device__ void __closesthit__plane()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    Vec3f local_n = Vec3f(0, 1, 0);
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);
    Vec2f uv = getVec2fFromAttribute<0>();

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = uv;
    si->surface_info = data->surface_info;

    si->shading.dpdu = optixTransformNormalFromObjectToWorldSpace(Vec3f(1, 0, 0));
    si->shading.dpdv = optixTransformNormalFromObjectToWorldSpace(Vec3f(0, 0, 1));
}

extern "C" __device__ float __continuation_callable__pdf_plane(
    const AreaEmitterInfo& area_info, const Vec3f & origin, const Vec3f & direction)
{
    const auto* plane = reinterpret_cast<Plane::Data*>(area_info.shape_data);

    SurfaceInteraction si;
    const Vec3f local_o = area_info.worldToObj.pointMul(origin);
    const Vec3f local_d = area_info.worldToObj.vectorMul(direction);

    if (!hitPlane(plane, local_o, local_d, 0.01f, 1e16f, si))
        return 0.0f;

    const Vec3f corner0 = area_info.objToWorld.pointMul(Vec3f(plane->min.x(), 0.0f, plane->min.y()));
    const Vec3f corner1 = area_info.objToWorld.pointMul(Vec3f(plane->max.x(), 0.0f, plane->min.y()));
    const Vec3f corner2 = area_info.objToWorld.pointMul(Vec3f(plane->min.x(), 0.0f, plane->max.y()));
    si.shading.n = normalize(area_info.objToWorld.vectorMul(si.shading.n));
    const float area = length(cross(corner1 - corner0, corner2 - corner0));
    const float distance_squared = si.t * si.t;
    const float cosine = fabs(dot(si.shading.n, direction));
    if (cosine < math::eps)
        return 0.0f;
    return distance_squared / (cosine * area);
}

// Return light vector in global space from si.p to random light point
extern "C" __device__ Vec3f __direct_callable__rnd_sample_plane(AreaEmitterInfo area_info, SurfaceInteraction * si)
{
    const auto* plane = reinterpret_cast<Plane::Data*>(area_info.shape_data);
    // Transform point from world to object space
    const Vec3f local_p = area_info.worldToObj.pointMul(si->p);
    uint32_t seed = si->seed;
    // Get random point on area emitter
    const Vec3f rnd_p(rnd(seed, plane->min.x(), plane->max.x()), 0.0f, rnd(seed, plane->min.y(), plane->max.y()));
    Vec3f to_light = rnd_p - local_p;
    to_light = area_info.objToWorld.vectorMul(to_light);
    si->seed = seed;
    return to_light;
}

// Sphere -------------------------------------------------------------------------------
static __forceinline__ __device__ bool hitSphere(const Sphere::Data* sphere_data, const Vec3f& o, const Vec3f& v, const float tmin, const float tmax, SurfaceInteraction& si)
{
    const Vec3f center = sphere_data->center;
    const float radius = sphere_data->radius;

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
    si.shading.uv = getSphereUV(si.shading.n);
    return true;
}

extern "C" __device__ void __intersection__sphere() {
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

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
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = getSphereUV(local_n);
    si->surface_info = data->surface_info;

    float phi = atan2(local_n.z(), local_n.x());
    if (phi < 0) phi += 2.0f * math::pi;
    const float theta = acos(local_n.y());
    const float u = phi / (2.0f * math::pi);
    const float v = theta / math::pi;
    const Vec3f dpdu = Vec3f(-math::two_pi * local_n.z(), 0, math::two_pi * local_n.x());
    const Vec3f dpdv = math::pi * Vec3f(local_n.y() * cos(phi), -sin(theta), local_n.y() * sin(phi));
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
}

// Cylinder -------------------------------------------------------------------------------
static INLINE DEVICE Vec2f getCylinderUV(
    const Vec3f& p, const float radius, const float height, const bool hit_disk)
{
    if (hit_disk)
    {
        const float r = sqrtf(p.x()*p.x() + p.z()*p.z()) / radius;
        const float theta = atan2(p.z(), p.x());
        float u = 1.0f - (theta + math::pi / 2.0f) * math::inv_pi;
        return Vec2f(u, r);
    } 
    else
    {
        float phi = atan2(p.z(), p.x());
        if (phi < 0.0f) phi += math::two_pi;
        const float u = phi / math::two_pi;
        const float v = (p.y() + height / 2.0f) / height;
        return Vec2f(u, v);
    }
}

extern "C" __device__ void __intersection__cylinder()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* cylinder = reinterpret_cast<Cylinder::Data*>(data->shape_data);

    const float radius = cylinder->radius;
    const float height = cylinder->height;

    Ray ray = getLocalRay();
    SurfaceInteraction* si = getSurfaceInteraction();
    
    const float a = dot(ray.d, ray.d) - ray.d.y() * ray.d.y();
    const float half_b = (ray.o.x() * ray.d.x() + ray.o.z() * ray.d.z());
    const float c = dot(ray.o, ray.o) - ray.o.y() * ray.o.y() - radius*radius;
    const float discriminant = half_b*half_b - a*c;

    if (discriminant > 0.0f)
    {
        const float sqrtd = sqrtf(discriminant);
        const float side_t1 = (-half_b - sqrtd) / a;
        const float side_t2 = (-half_b + sqrtd) / a;

        const float side_tmin = fmin( side_t1, side_t2 );
        const float side_tmax = fmax( side_t1, side_t2 );

        if ( side_tmin > ray.tmax || side_tmax < ray.tmin )
            return;

        const float upper = height / 2.0f;
        const float lower = -height / 2.0f;
        const float y_tmin = fmin( (lower - ray.o.y()) / ray.d.y(), (upper - ray.o.y()) / ray.d.y() );
        const float y_tmax = fmax( (lower - ray.o.y()) / ray.d.y(), (upper - ray.o.y()) / ray.d.y() );

        float t1 = fmax(y_tmin, side_tmin);
        float t2 = fmin(y_tmax, side_tmax);
        if (t1 > t2 || (t2 < ray.tmin) || (t1 > ray.tmax))
            return;
        
        bool check_second = true;
        if (ray.tmin < t1 && t1 < ray.tmax)
        {
            Vec3f P = ray.at(t1);
            bool hit_disk = y_tmin > side_tmin;
            Vec3f normal = hit_disk 
                         ? normalize(P - Vec3f(P.x(), 0.0f, P.z()))   // Hit at disk
                         : normalize(P - Vec3f(0.0f, P.y(), 0.0f));   // Hit at side
            Vec2f uv = getCylinderUV(P, radius, height, hit_disk);
            if (hit_disk)
            {
                const float rHit = sqrtf(P.x()*P.x() + P.z()*P.z());
                si->shading.dpdu = Vec3f(-math::two_pi * P.y(), 0.0f, math::two_pi * P.z());
                si->shading.dpdv = Vec3f(P.x(), 0.0f, P.z()) * radius / rHit;
            }
            else 
            {
                si->shading.dpdu = Vec3f(-math::two_pi * P.z(), 0.0f, math::two_pi * P.x());
                si->shading.dpdv = Vec3f(0.0f, height, 0.0f);
            }
            optixReportIntersection(t1, 0, Vec3f_as_ints(normal), Vec2f_as_ints(uv));
            check_second = false;
        }
        
        if (check_second)
        {
            if (ray.tmin < t2 && t2 < ray.tmax)
            {
                Vec3f P = ray.at(t2);
                bool hit_disk = y_tmax < side_tmax;
                Vec3f normal = hit_disk
                            ? normalize(P - Vec3f(P.x(), 0.0f, P.z()))   // Hit at disk
                            : normalize(P - Vec3f(0.0f, P.y(), 0.0f));   // Hit at side
                Vec2f uv = getCylinderUV(P, radius, height, hit_disk);
                if (hit_disk)
                {
                    const float rHit = sqrtf(P.x()*P.x() + P.z()*P.z());
                    si->shading.dpdu = Vec3f(-math::two_pi * P.y(), 0.0f, math::two_pi * P.z());
                    si->shading.dpdv = Vec3f(P.x(), 0.0f, P.z()) * radius / rHit;
                }
                else 
                {
                    si->shading.dpdu = Vec3f(-math::two_pi * P.z(), 0.0f, math::two_pi * P.x());
                    si->shading.dpdv = Vec3f(0.0f, height, 0.0f);
                }
                optixReportIntersection(t2, 0, Vec3f_as_ints(normal), Vec2f_as_ints(uv));
            }
        }
    }
}

extern "C" __device__ void __closesthit__cylinder()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* cylinder = reinterpret_cast<Cylinder::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec2f uv = getVec2fFromAttribute<3>();

    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = normalize(world_n);
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = uv;
    si->surface_info = data->surface_info;

    // dpdu and dpdv are calculated in intersection shader
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(si->shading.dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(si->shading.dpdv));
}

// Triangle mesh -------------------------------------------------------------------------------
extern "C" __device__ void __closesthit__mesh()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* mesh_data = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Shading shading = getMeshShading(mesh_data, optixGetTriangleBarycentrics(), optixGetPrimitiveIndex());

    // Linear interpolation of normal by barycentric coordinates.
    shading.n = normalize(optixTransformNormalFromObjectToWorldSpace(shading.n));
    shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdu));
    shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdv));

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}

