#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

struct SurfaceInteraction {
    Vec3f p; 
    Vec3f n;
    Vec3f albedo;
    Vec3f radiance;
    Vec2f uv;
};

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

INLINE DEVICE void trace(
    OptixTraversableHandle handle, Vec3f ro, Vec3f rd, 
    float tmin, float tmax, float ray_time, SurfaceInteraction* si)
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle, 
        ro, rd, tmin, tmax,
        ray_time,   // ray time
        OptixVisibilityMask(1), 
        OPTIX_RAY_FLAG_NONE, 
        0, 1, 0, 
        u0, u1      // payloads 
    );
}

// Raygen ----------------------------------------------------------------
extern "C" __device__ void __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const Vec3ui idx(optixGetLaunchIndex());

    Vec3f result(0.0f);

    uint32_t seed = tea<4>(idx.y() * params.width + idx.x(), params.frame);

    for (int i = 0; i < params.samples_per_launch; i++)
    {
        SurfaceInteraction si;
        const Vec2f subpixel_jitter = UniformSampler::get2D(seed) - 0.5f;

        const Vec2f res(params.width, params.height);
        const Vec2f d = 2.0f * ((Vec2f(idx.x(), idx.y()) + subpixel_jitter) / res) - 1.0f;
        Vec3f ro, rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

        trace(params.handle, ro, rd, 0.01f, 1e16f, rnd(seed), &si);

        result += si.radiance;
    }

    const uint32_t image_idx = idx.y() * params.width + idx.x();

    Vec3f accum_color = result / static_cast<float>(params.samples_per_launch);
    if (params.frame > 0)
    {
        const float a = 1.0f / static_cast<float>(params.frame + 1);
        const Vec3f accum_color_prev(params.accum_buffer[image_idx]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    
    Vec3u color = make_color(accum_color);
    params.result_buffer[image_idx] = Vec4u(color, 255);
    params.accum_buffer[image_idx] = Vec4f(accum_color, 1.0f);
}

// Miss ----------------------------------------------------------------
extern "C" __device__ void __miss__envmap()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    const auto* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
    auto* si = getSurfaceInteraction();

    Ray ray = getWorldRay();

    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f*1e8f;
    const float discriminant = half_b * half_b - a*c;

    float sqrtd = sqrtf(discriminant);
    float t = (-half_b + sqrtd) / a;

    Vec3f p = normalize(ray.at(t));

    float phi = atan2(p.z(), p.x());
    float theta = asin(p.y());
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) * math::inv_pi;
    
    si->uv = Vec2f(u, v);
    si->n = Vec3f(0.0f);
    si->p = p;
    Vec3f color = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data);
    si->radiance = color;
    si->albedo = color;
}

// Hitgroups ----------------------------------------------------------------
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

    Vec3f local_n(0, 1, 0);
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);
    Vec2f uv = getVec2fFromAttribute<0>();

    auto* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->n = faceforward(world_n, -ray.d, world_n);
    si->uv = uv;
    Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        data->texture.prg_id, si, data->texture.data);
    si->albedo = albedo;

    const Vec3f light_dir = normalize(params.light.pos - si->p);
    const Vec3f shade = params.light.intensity * params.light.color * fmaxf(0.0f, dot(light_dir, si->n)) * albedo;
    si->radiance = 0.8f * shade + 0.2f * albedo;
}

extern "C" __device__ void __closesthit__mesh()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const auto* mesh = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    Ray ray = getWorldRay();
    
    const int prim_id = optixGetPrimitiveIndex();
    const Face face = mesh->faces[prim_id];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const Vec2f texcoord0 = mesh->texcoords[face.texcoord_id.x()];
    const Vec2f texcoord1 = mesh->texcoords[face.texcoord_id.y()];
    const Vec2f texcoord2 = mesh->texcoords[face.texcoord_id.z()];
    const Vec2f texcoords = (1-u-v)*texcoord0 + u*texcoord1 + v*texcoord2;

    Vec3f n0 = normalize(mesh->normals[face.normal_id.x()]);
	Vec3f n1 = normalize(mesh->normals[face.normal_id.y()]);
	Vec3f n2 = normalize(mesh->normals[face.normal_id.z()]);

    // Linear interpolation of normal by barycentric coordinates.
    Vec3f local_n = (1.0f-u-v)*n0 + u*n1 + v*n2;
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->n = faceforward(world_n, -ray.d, world_n);
    si->uv = texcoords;
    Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        data->texture.prg_id, si, data->texture.data);
    si->albedo = albedo;

    const Vec3f light_dir = normalize(params.light.pos - si->p);
    const Vec3f shade = params.light.intensity * params.light.color * fmaxf(0.0f, dot(light_dir, si->n)) * albedo;
    si->radiance = 0.8f * shade + 0.2f * albedo;
}

static __forceinline__ __device__ Vec2f getUV(const Vec3f& p) {
    const float phi = atan2(p.z(), p.x());
    const float theta = asin(p.y());
    const float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    const float v = 1.0f - (theta + math::pi / 2.0f) * math::inv_pi;
    return Vec2f(u, v);
}

extern "C" __device__ void __intersection__sphere() {
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

extern "C" __device__ void __closesthit__sphere() {
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->n = faceforward(world_n, -ray.d, world_n);
    si->uv = getUV(local_n);
    Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        data->texture.prg_id, si, data->texture.data
    );
    si->albedo = albedo;
    
    const Vec3f light_dir = normalize(params.light.pos - si->p);
    const Vec3f shade = params.light.intensity * params.light.color * fmaxf(0.0f, dot(light_dir, si->n)) * albedo;
    si->radiance = 0.8f * shade + 0.2f * albedo;
}

// Textures ----------------------------------------------------------------
extern "C" __device__ Vec3f __direct_callable__bitmap(SurfaceInteraction* si, void* tex_data) {
    const auto* image = reinterpret_cast<BitmapTexture::Data*>(tex_data);
    float4 c = tex2D<float4>(image->texture, si->uv.x(), si->uv.y());
    return Vec3f(c);
}

extern "C" __device__ Vec3f __direct_callable__constant(SurfaceInteraction* si, void* tex_data) {
    const auto* constant = reinterpret_cast<ConstantTexture::Data*>(tex_data);
    return constant->color;
}

extern "C" __device__ Vec3f __direct_callable__checker(SurfaceInteraction* si, void* tex_data) {
    const auto* checker = reinterpret_cast<CheckerTexture::Data*>(tex_data);
    const bool is_odd = sinf(si->uv.x() * math::pi * checker->scale) * sinf(si->uv.y() * math::pi * checker->scale) < 0;
    return lerp(checker->color1, checker->color2, (float)is_odd);
}