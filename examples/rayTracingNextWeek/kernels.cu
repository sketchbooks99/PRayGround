#include <prayground/prayground.h>
#include "params.h"

// Utilities 

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

static INLINE DEVICE void trace(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd, 
    float tmin, float tmax, float ray_time, uint32_t ray_type, SurfaceInteraction* si)
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle, ro.toCUVec(), rd.toCUVec(), 
        tmin, tmax, ray_time, 
        OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 
        0, 1, 0, 
        u0, u1);
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

extern "C" __device__ void __raygen__pinhole()
{
    const RaygenData* raygen = (RaygenData*)optixGetSbtDataPointer();

    const int frame = params.frame;

    const Vec3ui idx(optixGetLaunchIndex());
    uint32_t seed = tea<4>(idx.y() * params.width + idx.x(), frame);

    Vec3f result(0.0f);
    Vec3f normal(0.0f);

    int i = params.samples_per_launch;

    do
    {
        const Vec2f subpixel_jitter = UniformSampler::get2D(seed) - 0.5f;

        const Vec2f d = 2.0f * make_float2(
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

            trace(params.handle, ro, rd, 0.01f, tmax, rnd(seed), /* ray_type = */ 0, &si);

            if (si.trace_terminate) {
                result += si.emission * throughput;
                break;
            }

            if (depth == 0)
                normal = si.shading.n;

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
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id, &si, si.surface_info.data);
                
                // Evaluate bsdf
                Vec3f bsdf_val = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id, &si, si.surface_info.data);
                throughput *= bsdf_val;
            }
            // Rough surface sampling with applying MIS
            else if ( +(si.surface_info.type & (SurfaceType::Rough | SurfaceType::Diffuse)) )
            {
                // Importance sampling according to the BSDF
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id, &si, si.surface_info.data);

                // Evaluate PDF of area emitter
                float pdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                    si.surface_info.pdf_id, &si, si.surface_info.data);

                // Evaluate BSDF
                Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id, &si, si.surface_info.data);

                throughput *= bsdf / pdf;
            }

            // プライマリーレイ以外ではtmaxは大きくしておく
            tmax = 1e16f;
            
            ro = si.p;
            rd = si.wi;

            ++depth;
        }
    } while (--i);

    const uint32_t image_index = idx.y() * params.width + idx.x();

    if (result.x() != result.x()) result.x() = 0.0f;
    if (result.y() != result.y()) result.y() = 0.0f;
    if (result.z() != result.z()) result.z() = 0.0f;

    Vec3f accum_color = result / static_cast<float>(params.samples_per_launch);

    if (frame > 0)
    {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_color_prev(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = Vec4f(accum_color, 1.0f);
    Vec3u color = make_color(reinhardToneMap(accum_color, params.white));
    params.result_buffer[image_index] = Vec4u(color, 255);
}

// Surfaces ------------------------------------------------------------------------------------------
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
    si->uv = Vec2f(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data);
}

// Hitgroups ------------------------------------------------------------------------------------------
// Plane
extern "C" __device__ void __intersection__plane()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Plane::Data* plane_data = reinterpret_cast<Plane::Data*>(data->shape_data);

    const Vec2f min = plane_data->min;
    const Vec2f max = plane_data->max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y / ray.d.y;

    const float x = ray.o.x + t * ray.d.x;
    const float z = ray.o.z + t * ray.d.z;

    Vec2f uv(x / (max.x() - min.x()), z / (max.y() - min.y()));

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, Vec2f_as_ints(uv));
}

extern "C" __device__ void __closesthit__plane()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    Vec3f local_n(0, 1, 0);
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);
    Vec2f uv = getVec2fFromAttribute<0>();

    auto* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;

    si->shading.dpdu = optixTransformNormalFromObjectToWorldSpace(Vec3f(1, 0, 0));
    si->shading.dpdv = optixTransformNormalFromObjectToWorldSpace(Vec3f(0, 0, 1));
}

// Sphere -------------------------------------------------------------------------------
static __forceinline__ __device__ Vec2f getSphereUV(const Vec3f& p) {
    float phi = atan2(p.z(), p.x());
    if (phi < 0) phi += math::two_pi;
    float theta = acos(p.y());
    float u = phi / math::two_pi;
    float v = theta / math::pi;
    return Vec2f(u, v);
}

static __forceinline__ __device__ bool hitSphere(
    const Sphere::Data* sphere_data, const Vec3f& o, const Vec3f& v,
    const float tmin, const float tmax, SurfaceInteraction* si)
{
    const Vec3f center = sphere_data->center;
    const float radius = sphere_data->radius;

    const Vec3f oc = o - center;
    const float a = dot(v, v);
    const float half_b = dot(oc, v);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant <= 0.0f) 
        return false;

    const float sqrtd = sqrtf(discriminant);

    float t = (-half_b - sqrtd) / a;
    if (t < tmin || tmax < t)
    {
        t = (-half_b + sqrtd) / a;
        if (t < tmin || tmax < t)
            return false;
    }

    si->p = o + t * v;
    si->shading.n = (si->p - center) / radius;
    si->t = t;
    si->uv = getSphereUV(si->shading.n);
    return true;
}

extern "C" __device__ void __intersection__sphere()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const int prim_id = optixGetPrimitiveIndex();
    Sphere::Data sphere_data = (reinterpret_cast<Sphere::Data*>(data->shape_data))[prim_id];

    Ray ray = getLocalRay();

    SurfaceInteraction si;
    if (hitSphere(&sphere_data, ray.o, ray.d, ray.tmin, ray.tmax, &si))
        optixReportIntersection(si.t, 0, Vec3f_as_ints(si.shading.n), Vec2f_as_ints(si.uv));
}

/// From "Ray Tracing: The Next Week" by Peter Shirley
/// @ref: https://raytracing.github.io/books/RayTracingTheNextWeek.html
extern "C" __device__ void __intersection__sphere_medium()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const int prim_id = optixGetPrimitiveIndex();
    SphereMedium::Data sphere_medium_data = reinterpret_cast<SphereMedium::Data*>(data->shape_data)[prim_id];
    SphereData sphere_data;
    sphere_data.center = sphere_medium_data.center;
    sphere_data.radius = sphere_medium_data.radius;
    const float density = sphere_medium_data.density;

    Ray ray = getLocalRay();

    SurfaceInteraction* global_si = getSurfaceInteraction();
    unsigned int seed = global_si->seed;

    SurfaceInteraction si1, si2;
    if (!hitSphere(&sphere_data, ray.o, ray.d, -1e16f, 1e16f, &si1))
        return;
    if (!hitSphere(&sphere_data, ray.o, ray.d, si1.t + math::eps, 1e16f, &si2))
        return;

    if (si1.t < ray.tmin) si1.t = ray.tmin;
    if (si2.t > ray.tmax) si2.t = ray.tmax;

    if (si1.t >= si2.t)
        return;

    if (si1.t < 0.0f)
        si1.t = 0.0f;

    const float neg_inv_density = -1.0f / sphere_medium_data.density;
    const float ray_length = length(ray.d);
    const float distance_inside_boundary = (si2.t - si1.t) * ray_length;
    const float hit_distance = neg_inv_density * logf(rnd(seed));
    global_si->seed = seed;

    if (hit_distance > distance_inside_boundary)
        return;

    const float t = si1.t + hit_distance / ray_length;
    optixReportIntersection(t, 0, float3_as_ints(si1.n), float2_as_ints(si1.uv));
}

extern "C" __device__ void __closesthit__sphere() {
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    float3 local_n = getFloat3FromAttribute<0>();
    float2 uv = getFloat2FromAttribute<3>();
    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = getSphereUV(local_n);
    si->surface_info = data->surface_info;

    // Calculate partial derivative on texture coordinates
    float phi = atan2(local_n.z, local_n.x);
    if (phi < 0) phi += 2.0f * math::pi;
    const float theta = acos(local_n.y);
    const float3 dpdu = make_float3(-math::two_pi * local_n.z, 0, math::two_pi * local_n.x);
    const float3 dpdv = math::pi * make_float3(local_n.y * cos(phi), -sin(theta), local_n.y * sin(phi));
    si->dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
}

// Box -------------------------------------------------------------------------------
static INLINE DEVICE float2 getBoxUV(const float3& p, const float3& min, const float3& max, const int axis)
{
    float2 uv;
    int u_axis = (axis + 1) % 3;
    int v_axis = (axis + 2) % 3;

    // axisがYの時は (u: Z, v: X) -> (u: X, v: Z)へ順番を変える
    if (axis == 1) swap(u_axis, v_axis);

    uv.x = (getByIndex(p, u_axis) - getByIndex(min, u_axis)) / (getByIndex(max, u_axis) - getByIndex(min, u_axis));
    uv.y = (getByIndex(p, v_axis) - getByIndex(min, v_axis)) / (getByIndex(max, v_axis) - getByIndex(min, v_axis));

    return clamp(uv, 0.0f, 1.0f);
}

static INLINE DEVICE int hitBox(
    const BoxData* box_data,
    const float3& o, const float3& v,
    const float tmin, const float tmax,
    SurfaceInteraction& si)
{
    float3 min = box_data->min;
    float3 max = box_data->max;

    float _tmin = tmin, _tmax = tmax;
    int min_axis = -1, max_axis = -1;

    for (int i = 0; i < 3; i++)
    {
        float t0, t1;
        if (getByIndex(v, i) == 0.0f)
        {
            t0 = fminf(getByIndex(min, i) - getByIndex(o, i), getByIndex(max, i) - getByIndex(o, i));
            t1 = fmaxf(getByIndex(min, i) - getByIndex(o, i), getByIndex(max, i) - getByIndex(o, i));
        }
        else
        {
            t0 = fminf((getByIndex(min, i) - getByIndex(o, i)) / getByIndex(v, i),
                (getByIndex(max, i) - getByIndex(o, i)) / getByIndex(v, i));
            t1 = fmaxf((getByIndex(min, i) - getByIndex(o, i)) / getByIndex(v, i),
                (getByIndex(max, i) - getByIndex(o, i)) / getByIndex(v, i));
        }
        min_axis = t0 > _tmin ? i : min_axis;
        max_axis = t1 < _tmax ? i : max_axis;

        _tmin = fmaxf(t0, _tmin);
        _tmax = fminf(t1, _tmax);

        if (_tmax < _tmin)
            return -1;
    }

    float3 center = (min + max) / 2.0f;
    if ((tmin < _tmin && _tmin < tmax) && (-1 < min_axis && min_axis < 3))
    {
        float3 p = o + _tmin * v;
        float3 center_axis = p;
        setByIndex(center_axis, min_axis, getByIndex(center, min_axis));
        float3 normal = normalize(p - center_axis);
        float2 uv = getBoxUV(p, min, max, min_axis);
        si.p = p;
        si.n = normal;
        si.uv = uv;
        si.t = _tmin;
        return min_axis;
    }

    if ((tmin < _tmax && _tmax < tmax) && (-1 < max_axis && max_axis < 3))
    {
        float3 p = o + _tmax * v;
        float3 center_axis = p;
        setByIndex(center_axis, max_axis, getByIndex(center, max_axis));
        float3 normal = normalize(p - center_axis);
        float2 uv = getBoxUV(p, min, max, max_axis);
        si.p = p;
        si.n = normal;
        si.uv = uv;
        si.t = _tmax;
        return max_axis;
    }
    return -1;
}

extern "C" __device__ void __intersection__box()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const int prim_id = optixGetPrimitiveIndex();
    BoxData box_data = reinterpret_cast<BoxData*>(data->shape_data)[prim_id];

    Ray ray = getLocalRay();

    SurfaceInteraction si;
    int hit_axis = hitBox(&box_data, ray.o, ray.d, ray.tmin, ray.tmax, si);
    if (hit_axis >= 0) {
        optixReportIntersection(si.t, 0, float3_as_ints(si.n), float2_as_ints(si.uv), hit_axis);
    }
}

/// From "Ray Tracing: The Next Week" by Peter Shirley
/// @ref: https://raytracing.github.io/books/RayTracingTheNextWeek.html
extern "C" __device__ void __intersection__box_medium()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const int prim_id = optixGetPrimitiveIndex();
    BoxMediumData box_medium_data = reinterpret_cast<BoxMediumData*>(data->shape_data)[prim_id];
    BoxData box_data;
    box_data.min = box_medium_data.min;
    box_data.max = box_medium_data.max;

    Ray ray = getLocalRay();

    SurfaceInteraction* global_si = getSurfaceInteraction();
    unsigned int seed = global_si->seed;

    SurfaceInteraction si1, si2;
    if (hitBox(&box_data, ray.o, ray.d, -1e16f, 1e16f, si1) < 0)
        return;
    if (hitBox(&box_data, ray.o, ray.d, si1.t + math::eps, 1e16f, si2) < 0)
        return;

    if (si1.t < ray.tmin) si1.t = ray.tmin;
    if (si2.t > ray.tmax) si2.t = ray.tmax;

    if (si1.t >= si2.t)
        return;

    if (si1.t < 0.0f)
        si1.t = 0.0f;

    const float neg_inv_density = -1.0f / box_medium_data.density;
    const float ray_length = length(ray.d);
    const float distance_inside_boundary = (si2.t - si1.t) * ray_length;
    const float hit_distance = neg_inv_density * logf(rnd(seed));
    global_si->seed = seed;

    if (hit_distance > distance_inside_boundary)
        return;

    const float t = si1.t + hit_distance / ray_length;
    optixReportIntersection(t, 0, float3_as_ints(si1.n), float2_as_ints(si1.uv), 0);
}

extern "C" __device__ void __closesthit__box()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    float3 local_n = make_float3(
        int_as_float(optixGetAttribute_0()),
        int_as_float(optixGetAttribute_1()),
        int_as_float(optixGetAttribute_2())
    );
    float2 uv = make_float2(
        int_as_float(optixGetAttribute_3()),
        int_as_float(optixGetAttribute_4())
    );

    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;

    uint32_t hit_axis = getAttribute<5>();
    float3 dpdu, dpdv;
    // x
    if (hit_axis == 0)
    {
        dpdu = make_float3(0.0f, 0.0f, 1.0f);
        dpdv = make_float3(0.0f, 1.0f, 0.0f);
    }
    else if (hit_axis == 1)
    {
        dpdu = make_float3(1.0f, 0.0f, 0.0f);
        dpdv = make_float3(0.0f, 0.0f, 1.0f);
    }
    else if (hit_axis == 2)
    {
        dpdu = make_float3(1.0f, 0.0f, 0.0f);
        dpdv = make_float3(0.0f, 1.0f, 0.0f);
    }
    si->dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
}

// Triangle mesh -------------------------------------------------------------------------------
extern "C" __device__ void __closesthit__mesh()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const MeshData* mesh_data = reinterpret_cast<MeshData*>(data->shape_data);

    Ray ray = getWorldRay();

    const int prim_id = optixGetPrimitiveIndex();
    const Face face = mesh_data->faces[prim_id];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float3 p0 = mesh_data->vertices[face.vertex_id.x];
    const float3 p1 = mesh_data->vertices[face.vertex_id.y];
    const float3 p2 = mesh_data->vertices[face.vertex_id.z];

    const float2 texcoord0 = mesh_data->texcoords[face.texcoord_id.x];
    const float2 texcoord1 = mesh_data->texcoords[face.texcoord_id.y];
    const float2 texcoord2 = mesh_data->texcoords[face.texcoord_id.z];
    const float2 texcoords = (1 - u - v) * texcoord0 + u * texcoord1 + v * texcoord2;

    float3 n0 = mesh_data->normals[face.normal_id.x];
    float3 n1 = mesh_data->normals[face.normal_id.y];
    float3 n2 = mesh_data->normals[face.normal_id.z];

    // Linear interpolation of normal by barycentric coordinates.
    float3 local_n = (1.0f - u - v) * n0 + u * n1 + v * n2;
    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = texcoords;
    si->surface_info = data->surface_info;

    // Calculate partial derivative on texture coordinates
    float3 dpdu, dpdv;
    const float2 duv02 = texcoord0 - texcoord2, duv12 = texcoord1 - texcoord2;
    const float3 dp02 = p0 - p2, dp12 = p1 - p2;
    const float D = duv02.x * duv12.y - duv02.y * duv12.x;
    bool degenerateUV = abs(D) < 1e-8f;
    if (!degenerateUV)
    {
        const float invD = 1.0f / D;
        dpdu = (duv12.y * dp02 - duv02.y * dp12) * invD;
        dpdv = (-duv12.x * dp02 + duv02.x * dp12) * invD;
    }
    if (degenerateUV || length(cross(dpdu, dpdv)) == 0.0f)
    {
        /// @note Is it OK with n = world_n? 
        const float3 n = normalize(cross(p2 - p0, p1 - p0));
        Onb onb(n);
        dpdu = onb.tangent;
        dpdv = onb.bitangent;
    }
    si->dpdu = normalize(optixTransformNormalFromObjectToWorldSpace(dpdu));
    si->dpdv = normalize(optixTransformNormalFromObjectToWorldSpace(dpdv));
}

// Surfaces ------------------------------------------------------------------------------------------
// Diffuse -----------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_diffuse(SurfaceInteraction* si, void* mat_data) {
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(mat_data);

    if (diffuse->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    si->trace_terminate = false;
    uint32_t seed = si->seed;
    Vec2f u = UniformSampler::get2D(seed);
    Vec3f wi = cosineSampleHemisphere(u[0], u[1]);
    Onb onb(si->shading.n);
    onb.inverseTransform(wi);
    si->wi = normalize(wi);
    si->seed = seed;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(mat_data);
    const Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(diffuse->texture.prg_id, si, diffuse->texture.data);
    si->albedo = albedo;
    si->emission = Vec3f(0.0f);
    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    return albedo * cosine * math::inv_pi;
}

extern "C" __device__ float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    return cosine * math::inv_pi;
}

// Dielectric --------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_dielectric(SurfaceInteraction * si, void* mat_data) {
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(mat_data);

    float ni = 1.0f; // air
    float nt = dielectric->ior;  // ior specified 
    float cosine = dot(si->wo, si->shading.n);
    bool into = cosine < 0;
    Vec3f outward_normal = into ? si->shading.n : -si->shading.n;

    if (!into) swap(ni, nt);

    cosine = fabs(cosine);
    float sine = sqrtf(1.0 - cosine * cosine);
    bool cannot_refract = (ni / nt) * sine > 1.0f;

    float reflect_prob = fresnel(cosine, ni, nt);
    unsigned int seed = si->seed;

    if (cannot_refract || reflect_prob > rnd(seed))
        si->wi = reflect(si->wo, outward_normal);
    else
        si->wi = refract(si->wo, outward_normal, cosine, ni, nt);
    si->radiance_evaled = false;
    si->trace_terminate = false;
    si->seed = seed;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_dielectric(SurfaceInteraction* si, void* mat_data)
{
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(mat_data);
    si->emission = Vec3f(0.0f);
    Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(dielectric->texture.prg_id, si, dielectric->texture.data);
    si->albedo = albedo;
    return albedo;
}

extern "C" __device__ float __direct_callable__pdf_dielectric(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}

// Conductor --------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_conductor(SurfaceInteraction* si, void* mat_data) {
    const Conductor::Data* conductor = reinterpret_cast<Conductor::Data*>(mat_data);
    if (conductor->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    si->wi = reflect(si->wo, si->shading.n);
    si->trace_terminate = false;
    si->radiance_evaled = false;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_conductor(SurfaceInteraction* si, void* mat_data)
{
    const Conductor::Data* conductor = reinterpret_cast<Conductor::Data*>(mat_data);
    si->emission = Vec3f(0.0f);
    Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(conductor->texture.prg_id, si, conductor->texture.data);
    si->albedo = albedo;
    return albedo;
}

extern "C" __device__ float __direct_callable__pdf_conductor(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}

// Disney BRDF ------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_disney(SurfaceInteraction* si, void* mat_data)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);

    if (disney->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    unsigned int seed = si->seed;
    Vec2f u = UniformSampler::get2D(seed);
    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    Onb onb(si->shading.n);

    if (rnd(seed) < diffuse_ratio)
    {
        Vec3f wi = cosineSampleHemisphere(u[0], u[1]);
        onb.inverseTransform(wi);
        si->wi = normalize(wi);
    }
    else
    {
        float gtr2_ratio = 1.0f / (1.0f + disney->clearcoat);
        Vec3f h;
        const float alpha = fmaxf(0.001f, disney->roughness);
        if (rnd(seed) < gtr2_ratio)
            h = sampleGGX(u[0], u[1], alpha);
        else
            h = sampleGTR1(u[0], u[1], alpha);
        onb.inverseTransform(h);
        si->wi = normalize(reflect(si->wo, h));
    }
    si->radiance_evaled = false;
    si->trace_terminate = false;
    si->seed = seed;
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
extern "C" __device__ Vec3f __continuation_callable__bsdf_disney(SurfaceInteraction * si, void* mat_data)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);
    si->emission = Vec3f(0.0f);

    const Vec3f V = -normalize(si->wo);
    const Vec3f L = normalize(si->wi);
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
    const float Fd90 = 0.5f + 2.0f * disney->roughness * LdotH * LdotH;
    const float FVd90 = fresnelSchlickT(NdotV, Fd90);
    const float FLd90 = fresnelSchlickT(NdotL, Fd90);
    const Vec3f f_diffuse = (base_color * math::inv_pi) * FVd90 * FLd90;

    // Subsurface
    const float Fss90 = disney->roughness * LdotH * LdotH;
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
    const float alpha_cc = lerp(0.1f, 0.001f, disney->clearcoat_gloss);
    const float Dcc = GTR1(NdotH, alpha_cc);
    const float Gcc = smithG_GGX(NdotV, 0.25f);
    const Vec3f f_clearcoat = Vec3f(0.25f * disney->clearcoat * (Fcc * Dcc * Gcc));

    const Vec3f out = (1.0f - disney->metallic) * (lerp(f_diffuse, f_subsurface, disney->subsurface) + f_sheen) + f_specular + f_clearcoat;
    return out * clamp(NdotL, 0.0f, 1.0f);
}

/**
 * @ref http://simon-kallweit.me/rendercompo2015/report/#adaptivesampling
 *
 * @todo Investigate correct evaluation of PDF.
 */
extern "C" __device__ float __direct_callable__pdf_disney(SurfaceInteraction * si, void* mat_data)
{
    const DisneyData* disney = reinterpret_cast<DisneyData*>(mat_data);

    const Vec3f V = -si->wo;
    const Vec3f L = si->wi;
    const Vec3f N = si->shading.n;

    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    const float specular_ratio = 1.0f - diffuse_ratio;

    const float NdotL = abs(dot(N, L));
    const float NdotV = abs(dot(N, V));

    const float alpha = fmaxf(0.001f, disney->roughness);
    const float alpha_cc = lerp(0.1f, 0.001f, disney->clearcoat_gloss);
    const Vec3f H = normalize(V + L);
    const float NdotH = abs(dot(H, N));

    const float pdf_Ds = GTR2(NdotH, alpha);
    const float pdf_Dcc = GTR1(NdotH, alpha_cc);
    const float ratio = 1.0f / (1.0f + disney->clearcoat);
    const float pdf_specular = (pdf_Dcc + ratio * (pdf_Ds - pdf_Dcc));
    const float pdf_diffuse = NdotL * math::inv_pi;

    return diffuse_ratio * pdf_diffuse + specular_ratio * pdf_specular;
}

// Isotropic ------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_isotropic(SurfaceInteraction* si, void* mat_data)
{
    const Isotropic::Data* iso = reinterpret_cast<Isotropic::Data*>(mat_data);
    uint32_t seed = si->seed;
    si->wi = normalize(Vec3f(rnd(seed, -1.0f, 1.0f), rnd(seed, -1.0f, 1.0f), rnd(seed, -1.0f, 1.0f)));
    si->trace_terminate = false;
    si->seed = seed;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_isotropic(SurfaceInteraction* si, void* mat_data)
{
    const Isotropic::Data* iso = reinterpret_cast<Isotropic::Data*>(mat_data);
    return iso->albedo / math::two_pi;
}

extern "C" __device__ float __direct_callable__pdf_isotropic(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f / math::two_pi;
}

// Area emitter ------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__area_emitter(SurfaceInteraction * si, void* area_data)
{
    const AreaEmitter::Data* area = reinterpret_cast<AreaEmitter::Data*>(area_data);
    si->trace_terminate = true;
    float is_emitted = 1.0f;
    if (!area->twosided)
        is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;

    const Vec3f base = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        area->texture.prg_id, si, area->texture.data);
    si->albedo = base;
    si->emission = base * area->intensity * is_emitted;
}

// Textures ----------------------------------------------------------------
extern "C" __device__ Vec3f __direct_callable__bitmap(SurfaceInteraction* si, void* tex_data) {
    const BitmapTexture::Data* image = reinterpret_cast<BitmapTexture::Data*>(tex_data);
    float4 c = tex2D<float4>(image->texture, si->uv.x(), si->uv.y());
    return Vec3f(c);
}

extern "C" __device__ Vec3f __direct_callable__constant(SurfaceInteraction* si, void* tex_data) {
    const ConstantTexture::Data* constant = reinterpret_cast<ConstantTexture::Data*>(tex_data);
    return constant->color;
}

extern "C" __device__ Vec3f __direct_callable__checker(SurfaceInteraction* si, void* tex_data) {
    const CheckerTexture::Data* checker = reinterpret_cast<CheckerTexture::Data*>(tex_data);
    const bool is_odd = sinf(si->uv.x() * math::pi * checker->scale) * sinf(si->uv.y() * math::pi * checker->scale) < 0;
    return lerp(checker->color1, checker->color2, (float)is_odd);
}