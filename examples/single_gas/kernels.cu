#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

static INLINE DEVICE void trace(
    OptixTraversableHandle handle, 
    const Vec3f& ro, const Vec3f& rd, 
    float tmin, float tmax,
    SurfaceInteraction* si)
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle, 
        ro, 
        rd, 
        tmin, 
        tmax, 
        0.0f, 
        OptixVisibilityMask(1), 
        OPTIX_RAY_FLAG_NONE, 
        0, 
        1, 
        0, 
        u0, u1);
}

// Raygen ----------------------------------------------------------------
extern "C" __device__ void __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const Vec3ui idx(optixGetLaunchIndex());
    Vec3f color(0.0f);
    
    SurfaceInteraction si;

    const Vec2f d = 2.0f * Vec2f((float)idx.x() / params.width, (float)idx.y() / params.height) - 1.0f;
    Vec3f ro, rd;
    getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);
    
    trace(params.handle, ro, rd, 0.01f, 1e16f, &si);

    color = si.albedo;

    const uint32_t image_index = idx.y() * params.width + idx.x();

    if (isnan(color.x()) || isinf(color.x())) color.x() = 0.0f;
    if (isnan(color.y()) || isinf(color.y())) color.y() = 0.0f;
    if (isnan(color.z()) || isinf(color.z())) color.z() = 0.0f;

    Vec3u result = make_color(color);
    params.result_buffer[image_index] = Vec4u(result, 255);
}

// Miss ----------------------------------------------------------------
extern "C" __device__ void __miss__envmap()
{
    const auto* data = (MissData*)optixGetSbtDataPointer();
    const auto* env = (EnvironmentEmitter::Data*)(data->env_data);
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
    float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    si->shading.uv = Vec2f(u, v);
    si->albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data
    );
}

// Hitgroups ----------------------------------------------------------------
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
    const auto* data = (HitgroupData*)(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    Vec3f local_n(0, 1, 0);
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n.toCUVec());
    world_n = normalize(world_n);
    Vec2f uv = getVec2fFromAttribute<0>();

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->shading.uv = uv;
    si->albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        data->texture.prg_id, si, data->texture.data);
}

extern "C" __device__ void __closesthit__mesh()
{
    const auto* data = (HitgroupData*)optixGetSbtDataPointer();
    const auto* mesh_data = (TriangleMesh::Data*)data->shape_data;

    Ray ray = getWorldRay();
    
    const int prim_id = optixGetPrimitiveIndex();
    const Face face = mesh_data->faces[prim_id];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const Vec2f texcoord0 = mesh_data->texcoords[face.texcoord_id.x()];
    const Vec2f texcoord1 = mesh_data->texcoords[face.texcoord_id.y()];
    const Vec2f texcoord2 = mesh_data->texcoords[face.texcoord_id.z()];
    const Vec2f texcoords = (1-u-v)*texcoord0 + u*texcoord1 + v*texcoord2;

    Vec3f n0 = normalize(mesh_data->normals[face.normal_id.x()]);
	Vec3f n1 = normalize(mesh_data->normals[face.normal_id.y()]);
	Vec3f n2 = normalize(mesh_data->normals[face.normal_id.z()]);

    // Linear interpolation of normal by barycentric coordinates.
    Vec3f local_n = (1.0f-u-v)*n0 + u*n1 + v*n2;
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n.toCUVec());
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->shading.uv = texcoords;
    si->albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        data->texture.prg_id, si, data->texture.data);
}

// Textures ----------------------------------------------------------------
extern "C" __device__ Vec3f __direct_callable__bitmap(SurfaceInteraction* si, void* tex_data) {
    const auto* image = reinterpret_cast<BitmapTexture::Data*>(tex_data);
    Vec4f c = tex2D<float4>(image->texture, si->shading.uv.x(), si->shading.uv.y());
    return Vec3f(c);
}

extern "C" __device__ Vec3f __direct_callable__constant(SurfaceInteraction* si, void* tex_data) {
    const auto* constant = reinterpret_cast<ConstantTexture::Data*>(tex_data);
    return constant->color;
}

extern "C" __device__ Vec3f __direct_callable__checker(SurfaceInteraction* si, void* tex_data) {
    const auto* checker = reinterpret_cast<CheckerTexture::Data*>(tex_data);
    const bool is_odd = sinf(si->shading.uv.x() * math::pi * checker->scale) * sinf(si->shading.uv.y() * math::pi * checker->scale) < 0;
    return is_odd ? checker->color1 : checker->color2;
}
