#include <prayground/prayground.h>

#include "params.h"

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

static __forceinline__ __device__ SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle, const Ray& ray, uint32_t ray_type, SurfaceInteraction* si
)
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle, ray.o, ray.d, ray.tmin, ray.tmax, 0, 
        OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 
        ray_type, 1, ray_type, 
        u0, u1
    );
}

extern "C" __device__ void __raygen__shade()
{
    const pgRaygenData<Camera>* raygen = reinterpret_cast<const pgRaygenData<Camera>*>(optixGetSbtDataPointer());

    const int32_t frame = params.frame;

    const Vec3ui idx(optixGetLaunchIndex());
    uint32_t seed = tea<4>(idx.y() * params.width + idx.x(), frame);

    Vec3f result(0.0f);
    SurfaceInteraction si;
    si.seed = seed;
    si.emission = 0.0f;
    si.albedo = 0.0f;
    si.trace_terminate = false;

    const Vec2f d = 2.0f * (Vec2f(idx.x(), idx.y()) / Vec2f(params.width, params.height)) - 1.0f;
    Vec3f ro, rd;
    getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

    Ray ray(ro, rd, 0.01f, 1e16f);
    trace(params.handle, ray, 0, &si);

    if (si.trace_terminate)
    {
        result = si.emission;
    }
    else
    {
        si.wi = normalize(Vec3f(-1, -1, 2));
        Vec3f shade = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
            si.surface_info.callable_id.bsdf, &si, si.surface_info.data);
        //Vec3f shade = dot(si.wi, si.shading.n) * Vec3f(0.3f, 0.7f, 0.3f);
        result = shade;
    }

    if (!result.isValid())
        result = Vec3f(0.0f);

    uint32_t image_idx = idx.y() * params.width + idx.x();
    params.result_buffer[image_idx] = make_color(Vec4f(result));
}

extern "C" __device__ void __miss__envmap()
{
    auto* miss = reinterpret_cast<const pgMissData*>(optixGetSbtDataPointer());
    auto* env = reinterpret_cast<const EnvironmentEmitter::Data*>(miss->env_data);
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
    si->emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data);
}

extern "C" __device__ void __closesthit__mesh()
{
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const auto* mesh = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);
    const uint32_t prim_idx = optixGetPrimitiveIndex();

    Ray ray = getWorldRay();

    const Vec2f bc = optixGetTriangleBarycentrics();
    Shading shading = getMeshShading(mesh, bc, prim_idx);
    shading.n = optixTransformNormalFromObjectToWorldSpace(shading.n);
    shading.dpdu = optixTransformVectorFromObjectToWorldSpace(shading.dpdu);
    shading.dpdv = optixTransformVectorFromObjectToWorldSpace(shading.dpdv);

    const Face f = mesh->faces[prim_idx];
    const Vec3f p0 = mesh->vertices[f.vertex_id.x()];
    const Vec3f p1 = mesh->vertices[f.vertex_id.y()];
    const Vec3f p2 = mesh->vertices[f.vertex_id.z()];
    const Vec3f n0 = mesh->normals[f.normal_id.x()];
    const Vec3f n1 = mesh->normals[f.normal_id.y()];
    const Vec3f n2 = mesh->normals[f.normal_id.z()];
    shading.n = (1.0f - bc.x() - bc.y()) * n0 + bc.x() * n1 + bc.y() * n2;
    shading.n = normalize(optixTransformNormalFromObjectToWorldSpace(shading.n));

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}

extern "C" __device__ void __closesthit__custom()
{
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
    // Transform shading frame to world space
    shading->n = normalize(optixTransformNormalFromObjectToWorldSpace(shading->n));
    shading->dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(shading->dpdu));
    shading->dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(shading->dpdv));

    auto* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading = *shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}

extern "C" __device__ Vec3f __direct_callable__phong(SurfaceInteraction* si, void* mat_data)
{
    const PhongData* phong = reinterpret_cast<const PhongData*>(mat_data);
    si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);
    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    return phong->diffuse * cosine + phong->ambient;
}

extern "C" __device__ Vec3f __direct_callable__brdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
}

extern "C" __device__ Vec3f __direct_callable__sampling_diffuse(SurfaceInteraction* si, void* mat_data)
{

}

extern "C" __device__ float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* mat_data)
{

}

// Texture func
extern "C" __device__ Vec3f __direct_callable__constant(SurfaceInteraction * si, void* tex_data)
{
    return getConstantTextureValue<Vec3f>(si->shading.uv, tex_data);
}

extern "C" __device__ Vec3f __direct_callable__checker(SurfaceInteraction * si, void* tex_data)
{
    return getCheckerTextureValue<Vec3f>(si->shading.uv, tex_data);
}

extern "C" __device__ Vec3f __direct_callable__bitmap(SurfaceInteraction * si, void* tex_data)
{
    return getBitmapTextureValue<Vec3f>(si->shading.uv, tex_data);
}