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
    shading->n = optixTransformNormalFromObjectToWorldSpace(shading->n);
    shading->dpdu = optixTransformVectorFromObjectToWorldSpace(shading->dpdu);
    shading->dpdv = optixTransformVectorFromObjectToWorldSpace(shading->dpdv);

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

    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    return phong->diffuse * cosine + phong->ambient;
}