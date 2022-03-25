#include <prayground/prayground.h>
#include "params.h"
#include <prayground/ext/nanovdb/util/Ray.h>
#include <prayground/ext/nanovdb/util/HDDA.h>
#include <prayground/ext/nanovdb/util/SampleFromVoxels.h>

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

static __forceinline__ __device__ SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = getPayload<0>();
    const unsigned int u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle,
    const Vec3f& ro, const Vec3f& rd,
    float tmin, float tmax,
    unsigned int ray_type,
    SurfaceInteraction* si
)
{
    unsigned int u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle,
        ro.toCUVec(), rd.toCUVec(),
        tmin, tmax, 0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        ray_type,
        1,
        ray_type,
        u0, u1
    );
}

/* Raygen function */
static __forceinline__ __device__ void getCameraRay(
    const Camera::Data& camera,
    const float& x, const float& y,
    Vec3f& ro, Vec3f& rd, uint32_t& seed)
{
    rd = normalize(x * camera.U + y * camera.V + camera.W);
    ro = camera.origin;
}

extern "C" __device__ void __raygen__medium()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int frame = params.frame;
    const Vec3ui idx = optixGetLaunchIndex();
    uint32_t seed = tea<4>(idx.x() * params.width + idx.y(), frame);

    Vec3f result = Vec3f(0.0f);
    Vec3f albedo = Vec3f(0.0f);

    int i = params.samples_per_launch;

    do
    {
        const Vec2f subpixel_jitter = Vec2f(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

        const Vec2f d = 2.0f * Vec2f(
            (static_cast<float>(idx.x()) + subpixel_jitter.x()) / static_cast<float>(params.width),
            (static_cast<float>(idx.y()) + subpixel_jitter.y()) / static_cast<float>(params.height)
        ) - 1.0f;

        Vec3f ro, rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd, seed);

        Vec3f throughput = Vec3f(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = Vec3f(0.0f);
        si.albedo = Vec3f(0.0f);
        si.trace_terminate = false;
        si.radiance_evaled = false;

        int depth = 0;
        for (;; ) {

            if (depth >= params.max_depth)
                break;

            trace(params.handle, ro, rd, 0.01f, 1e16f, 0, &si);

            if (si.trace_terminate) {
                result += si.emission * throughput;
                break;
            }

            if (si.surface_info.type == SurfaceType::AreaEmitter)
            {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id, &si, si.surface_info.data);
                result += si.emission * throughput;

                if (si.trace_terminate)
                    break;
            }
            // Material and medium sampling
            else if (+(si.surface_info.type & (SurfaceType::Material | SurfaceType::Medium)))
            {
                // Sampling scattered direction
                float pdf;
                Vec3f bsdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*, float&>(
                    si.surface_info.sample_id, &si, si.surface_info.data, pdf);
                throughput *= bsdf / pdf;
            }

            ro = si.p;
            rd = si.wi;

            ++depth;
        }
    } while (--i);

    const uint32_t image_index = idx.y() * params.width + idx.x();

    if (result[0] != result[0]) result[0] = 0.0f;
    if (result[1] != result[1]) result[1] = 0.0f;
    if (result[2] != result[2]) result[2] = 0.0f;

    Vec3f accum_color = result / static_cast<float>(params.samples_per_launch);

    if (frame > 0)
    {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_color_prev = Vec3f(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = Vec4f(accum_color, 1.0f);
    Vec3u color = make_color(accum_color, true);
    params.result_buffer[image_index] = Vec4u(color.x(), color.y(), color.z(), 255);
}

/* Miss function */
extern "C" __device__ void __miss__envmap()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    EnvironmentEmitter::Data* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
    SurfaceInteraction* si = getSurfaceInteraction();

    Ray ray = getWorldRay();
    const float lambda = __int_as_float(getPayload<2>());

    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f*1e8f;
    const float D = half_b * half_b - a * c;

    float sqrtD = sqrtf(D);
    float t = (-half_b + sqrtD) / a;

    const Vec3f p = normalize(ray.at(t));

    const float phi = atan2(p.z(), p.x());
    const float theta = asin(p.y());
    const float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    const float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    si->uv = Vec2f(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Vec3f, const Vec2f&, void*>(
        env->texture.prg_id, si->uv, env->texture.data
    );
    si->albedo = si->emission;
}

/* Material functions */
extern "C" __device__ Vec3f __direct_callable__sample_diffuse(SurfaceInteraction* si, void* mat_data, float& pdf) {
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(mat_data);

    if (diffuse->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    // Sampling scattering direction 
    si->trace_terminate = false;
    uint32_t seed = si->seed;
    const float z0 = rnd(seed);
    const float z1 = rnd(seed);
    Vec3f wi = cosineSampleHemisphere(z0, z1);
    Onb onb(si->shading.n);
    onb.inverseTransform(wi);
    si->wi = normalize(wi);
    si->seed = seed;

    // Get albedo from texture
    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(
        diffuse->texture.prg_id, si->uv, diffuse->texture.data);
    si->albedo = albedo;
    si->emission = Vec3f(0.0f);

    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    pdf = cosine / math::pi;
    return albedo * (cosine / math::pi);
}

extern "C" __device__ Vec3f __direct_callable__sample_medium(SurfaceInteraction* si, void* mat_data, float& pdf)
{
    const VDBGrid::Data* grid = reinterpret_cast<const VDBGrid::Data*>(mat_data);
    const float g = grid->g;
    const Vec3f sigma_s = grid->sigma_s;
    const float sigma_t = grid->sigma_t;
    si->trace_terminate = false;
    uint32_t seed = si->seed;
    Vec2f u = UniformSampler::get2D(seed);

    // Sampling scattering direction from Henyey--Greenstein phase function
    Vec3f wi = sampleHenyeyGreenstein(u, g);
    Onb onb(si->wo);
    onb.inverseTransform(wi);
    si->wi = wi;

    si->albedo = sigma_s;
    const float cos_theta = fmaxf(dot(si->wo, si->wi), 0.0f);
    const float phase = phaseHenyeyGreenstein(cos_theta, g);
    const Vec3f val = phase * (sigma_s / sigma_t);
    pdf = phase;
    return val;
}

extern "C" __device__ void __direct_callable__area_emitter(SurfaceInteraction* si, void* surface_data)
{
    const AreaEmitter::Data* area = reinterpret_cast<AreaEmitter::Data*>(surface_data);
    float is_emitted = 1.0f;
    if (!area->twosided)
        is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    
    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(
        area->texture.prg_id, si->uv, area->texture.data);
    si->albedo = base;
    si->emission = base * area->intensity * is_emitted;
    si->trace_terminate = true;
}

/* Hitgroup functions */
extern "C" __device__ void __intersection__plane()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Plane::Data* plane = reinterpret_cast<Plane::Data*>(data->shape_data);

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
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    Vec3f local_n = Vec3f(0, 1, 0);
    const Vec3f world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n.toCUVec()));
    const Vec2f uv = getVec2fFromAttribute<0>();

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

static __forceinline__ __device__ Vec2f getSphereUV(const Vec3f& p) {
    float phi = atan2(p.z(), p.x());
    float theta = asin(p.y());
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    return Vec2f(u, v);
}

extern "C" __device__ void __intersection__sphere()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

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

extern "C" __device__ void __closesthit__sphere()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Sphere::Data* sphere_data = reinterpret_cast<Sphere::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    const Vec3f world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n.toCUVec()));

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

static __forceinline__ __device__ float deltaTracking(
    const VDBGrid::Data* medium,
    const nanovdb::Vec3f& ro, const nanovdb::Vec3f& rd, 
    const float tmin, const float tmax, uint32_t& seed)
{
    const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(medium->density);
    assert(grid);

    const auto& tree = grid->tree();
    auto acc = tree.getAccessor();

    /// Extinction coefficient
    /// @todo Be sure to satisfy sigma_t >= density at x' (x': a point inside medium)
    const float sigma_t = medium->sigma_t;
    const Vec3f sigma_s = medium->sigma_s;
    using AccT = decltype(acc);
    nanovdb::SampleFromVoxels<AccT, 1, false> sample_from_voxels(acc);

    const auto ray = nanovdb::Ray<float>(ro, rd);

    float t = tmin;
    while (true)
    {
        Vec2f u = UniformSampler::get2D(seed);
        t -= logf(1.0f - u[0]) / sigma_t;
        if (t >= tmax) break;

        nanovdb::Coord ijk = nanovdb::RoundDown<nanovdb::Coord>(ray(t));
        const float density = sample_from_voxels(ijk) * sigma_s[0] * (1.0f - params.cloud_opacity);
        if (u[1] < (density / sigma_t))
            break;
    }

    return t;
}

/// Stochastic evaluation of transmittance.
/// @note Does this work under only monte-carlo simulation?
static __forceinline__ __device__ float deltaTrackingTransmittance(
    const VDBGrid::Data* medium, 
    const nanovdb::Vec3f& ro, const nanovdb::Vec3f& rd, 
    const float tmin, const float tmax, uint32_t& seed)
{
    const float t = deltaTracking(medium, ro, rd, tmin, tmax, seed);
    return static_cast<float>(t > tmax);
}

static __forceinline__ __device__ float ratioTrackingTransmittance(
    const VDBGrid::Data* medium, 
    const nanovdb::Vec3f& ro, const nanovdb::Vec3f& rd, 
    const float tmin, const float tmax, uint32_t& seed)
{
    const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(medium->density);
    assert(grid);

    const auto& tree = grid->tree();
    auto acc = tree.getAccessor();
    using AccT = decltype(acc);
    nanovdb::SampleFromVoxels<AccT, 1, false> sample_from_voxels(acc);

    const float sigma_t = medium->sigma_t;

    float Tr = 1.0f; // transmittance
    float t = tmin;

    const auto ray = nanovdb::Ray<float>(ro, rd, tmin, tmax);
    while (true)
    {
        t -= logf(1.0f - UniformSampler::get1D(seed)) / sigma_t;
        if (t > tmax) break;

        // Current position
        nanovdb::Coord ijk = nanovdb::RoundDown<nanovdb::Coord>(ray(t));
        // Get density from grid
        const float density = sample_from_voxels(ijk);
        // Update transmittance according to the ray
        Tr *= (1.0f - (density / sigma_t));
    }
    return Tr;
}    

extern "C" __device__ void __intersection__grid()
{
    const HitgroupData* data = reinterpret_cast<const HitgroupData*>(optixGetSbtDataPointer());
    const VDBGrid::Data* medium = reinterpret_cast<const VDBGrid::Data*>(data->shape_data);
    const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(medium->density);
    assert(grid);

    Ray ray = getLocalRay();

    nanovdb::Vec3f ro(ray.o[0], ray.o[1], ray.o[2]);
    nanovdb::Vec3f rd(ray.d[0], ray.d[1], ray.d[2]);

    float t0 = ray.tmin;
    float t1 = ray.tmax;
    nanovdb::Ray<float> iray(ro, rd, t0, t1);
    auto bbox = grid->indexBBox();

    auto* si = getSurfaceInteraction();
    uint32_t seed = si->seed;
    if (iray.intersects(bbox, t0, t1))
    {
        t0 = fmaxf(t0, ray.tmin);

        const float t = deltaTracking(medium, ro, rd, t0, t1, seed);

        if (t < t1) {
            si->seed = seed;
            optixReportIntersection(t, 0);
        }
    }
}

extern "C" __device__ void __closesthit__grid()
{
    const HitgroupData* data = reinterpret_cast<const HitgroupData*>(optixGetSbtDataPointer());
    const VDBGrid::Data* medium = reinterpret_cast<const VDBGrid::Data*>(data->shape_data);
    const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(medium->density);
    const auto& tree = grid->tree();
    auto acc = tree.getAccessor();

    const float sigma_t = medium->sigma_t;

    Ray ray = getWorldRay();
    auto* si = getSurfaceInteraction();
    
    si->t = ray.tmax;
    si->p = ray.at(ray.tmax);
    si->shading.n = Vec3f(0,1,0);   // arbitrary
    si->wo = ray.d;
    si->surface_info = SurfaceInfo{ 
        data->shape_data, 
        data->surface_info.sample_id, 
        data->surface_info.sample_id, 
        data->surface_info.sample_id, 
        SurfaceType::Medium 
    };
    si->uv = Vec2f(0.5f);           // arbitrary
}

/* Texture functions */
extern "C" __device__ Vec3f __direct_callable__bitmap(const Vec2f& uv, void* tex_data)
{
    const BitmapTexture::Data* image = reinterpret_cast<BitmapTexture::Data*>(tex_data);
    Vec4f c = tex2D<float4>(image->texture, uv.x(), uv.y());
    return Vec3f(c.x(), c.y(), c.z());
}

extern "C" __device__ Vec3f __direct_callable__constant(const Vec2f& uv, void* tex_data)
{
    const ConstantTexture::Data* constant = reinterpret_cast<ConstantTexture::Data*>(tex_data);
    return constant->color;
}

extern "C" __device__ Vec3f __direct_callable__checker(const Vec2f& uv, void* tex_data)
{
    const CheckerTexture::Data* checker = reinterpret_cast<CheckerTexture::Data*>(tex_data);
    const bool is_odd = sinf(uv.x() * math::pi * checker->scale) * sinf(uv.y() * math::pi * checker->scale) < 0;
    return is_odd ? checker->color1 : checker->color2;
}



