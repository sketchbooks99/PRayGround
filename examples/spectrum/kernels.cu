#include <prayground/optix/cuda/device_util.cuh>
#include <prayground/core/spectrum.h>
#include <prayground/core/ray.h>
#include <prayground/core/onb.h>
#include <prayground/math/random.h>

#include <prayground/material/dielectric.h>
#include <prayground/material/diffuse.h>
#include <prayground/material/disney.h>

#include <prayground/texture/bitmap.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>

#include <prayground/shape/trianglemesh.h>
#include <prayground/shape/plane.h>
#include <prayground/shape/sphere.h>

#include "params.h"

using namespace prayground;

#define SAMPLE_FUNC(name) __direct_callable__sample_ ## name
#define BSDF_FUNC(name) __continuation_callable__bsdf_ ## name
#define PDF_FUNC(name) __direct_callable__pdf_ ## name
#define USE_SAMPLED_SPECTRUM 1

#if USE_SAMPLED_SPECTRUM
using Spectrum = SampledSpectrum;
#else 
using Spectrum = RGBSpectrum;
#endif

extern "C" { __constant__ LaunchParams params; }

static INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

// Raygen function
extern "C" __global__ void __raygen__spectrum()
{

}

// Miss function
extern "C" __device__ void __miss__envmap()
{

}

// Material functions
extern "C" __device__ void SAMPLE_FUNC(dielectric)(float lambda, SurfaceInteraction* si, void* mat_data)
{

}

extern "C" __device__ Spectrum BSDF_FUNC(dielectric)(SurfaceInteraction* si, void* mat_data)
{
    const DielectricData* dielectric = reinterpret_cast<DielectricData*>(mat_data);
    si->emission = make_float3(0.0f);
    float4 albedo = optixDirectCall<Spectrum, SurfaceInteraction*, void*>(dielectric->tex_program_id, si, dielectric->tex_data);
}

extern "C" __device__ float PDF_FUNC(dielectric)(SurfaceInteraction * si, void* mat_data)
{

}

extern "C" __device__ void SAMPLE_FUNC(diffuse)(float lambda, SurfaceInteraction * si, void* mat_data)
{

}

extern "C" __device__ Spectrum BSDF_FUNC(diffuse)(SurfaceInteraction * si, void* mat_data)
{

}

extern "C" __device__ float PDF_FUNC(diffuse)(SurfaceInteraction * si, void* mat_data)
{

}

extern "C" __device__ void SAMPLE_FUNC(disney)(float lambda, SurfaceInteraction * si, void* mat_data)
{

}

extern "C" __device__ Spectrum BSDF_FUNC(disney)(SurfaceInteraction * si, void* mat_data)
{

}

extern "C" __device__ float PDF_FUNC(disney)(SurfaceInteraction * si, void* mat_data)
{

}

// Texture functions
extern "C" __device__ Spectrum DC_FUNC(constant)(SurfaceInteraction * si, void* tex_data)
{
    const BitmapTextureData* image = reinterpret_cast<BitmapTextureData*>(tex_data);
}

extern "C" __device__ Spectrum DC_FUNC(checker)(SurfaceInteraction * si, void* tex_data)
{

}

extern "C" __device__ Spectrum DC_FUNC(bitmap)(SurfaceInteraction * si, void* tex_data)
{

}

// Hitgroup functions
extern "C" __device__ void CH_FUNC(mesh)()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const MeshData* mesh = reinterpret_cast<MeshData*>(data->shape_data);

    Ray ray = getWorldRay();

    const int id = optixGetPrimitiveIndex();
    const Face face = mesh->faces[id];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float3 p0 = mesh->vertices[face.vertex_id.x];
    const float3 p1 = mesh->vertices[face.vertex_id.y];
    const float3 p2 = mesh->vertices[face.vertex_id.z];

    const float2 texcoord0 = mesh->texcoords[face.texcoord_id.x];
    const float2 texcoord1 = mesh->texcoords[face.texcoord_id.y];
    const float2 texcoord2 = mesh->texcoords[face.texcoord_id.z];
    const float2 texcoords = (1 - u - v) * texcoord0 + u * texcoord1 + v * texcoord2;

    const float3 n0 = mesh->normals[face.normal_id.x];
    const float3 n1 = mesh->normals[face.normal_id.y];
    const float3 n2 = mesh->normals[face.normal_id.z];

    const float3 local_n = (1 - u - v) * n0 + u * n1 + v * n2;
    const float3 world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n));

    auto si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = texcoords;
    si->surface_info = data->surface_info;

    float3 dpdu, dpdv;
    const float2 duv02 = texcoord0 - texcoord2;
    const float2 duv12 = texcoord1 - texcoord2;
    const float3 dp02 = p0 - p2;
    const float3 dp12 = p1 - p2;
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
        const float3 n = normalize(cross(p2 - p0, p1 - p0));
        Onb onb(n);
        dpdu = onb.tangent;
        dpdv = onb.bitangent;
    }
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
}

static __forceinline__ __device__ float2 getSphereUV(const float3& p) {
    float phi = atan2(p.z, p.x);
    float theta = asin(p.y);
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    return make_float2(u, v);
}

extern "C" __device__ void IS_FUNC(sphere)()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const SphereData* sphere = reinterpret_cast<SphereData*>(data->shape_data);

    const float3 center = sphere->center;
    const float radius = sphere->radius;
    
    Ray ray = getLocalRay();

    const float3 oc = ray.o - center;
    const float a = dot(ray.d, ray.d);
    const float half_b = dot(oc, ray.d);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant > 0.0f) {
        float sqrtd = sqrtf(discriminant);
        float t1 = (-half_b - sqrtd) / a;
        bool check_second = true;
        if (t1 > ray.tmin && t1 < ray.tmax) {
            float3 normal = normalize((ray.at(t1) - center) / radius);
            check_second = false;
            optixReportIntersection(t1, 0, float3_as_ints(normal));
        }

        if (check_second) {
            float t2 = (-half_b + sqrtd) / a;
            if (t2 > ray.tmin && t2 < ray.tmax) {
                float3 normal = normalize((ray.at(t2) - center) / radius);
                optixReportIntersection(t2, 0, float3_as_ints(normal));
            }
        }
    }
}

extern "C" __device__ void CH_FUNC(sphere)()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const SphereData* sphere_data = reinterpret_cast<SphereData*>(data->shape_data);

    Ray ray = getWorldRay();

    float3 local_n = getFloat3FromAttribute<0>();
    const float3 world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n));

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = getSphereUV(local_n);
    si->surface_info = data->surface_info;

    float phi = atan2(local_n.z, local_n.x);
    if (phi < 0) phi += 2.0f * math::pi;
    const float theta = acos(local_n.y);
    const float3 dpdu = make_float3(-math::two_pi * local_n.z, 0, math::two_pi * local_n.x);
    const float3 dpdv = math::pi * make_float3(local_n.y * cos(phi), -sin(theta), local_n.y * sin(phi));
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
}

extern "C" __device__ void IS_FUNC(plane)()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const PlaneData* plane_data = reinterpret_cast<PlaneData*>(data->shape_data);

    const float2 min = plane_data->min;
    const float2 max = plane_data->max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y / ray.d.y;

    const float x = ray.o.x + t * ray.d.x;
    const float z = ray.o.z + t * ray.d.z;

    float2 uv = make_float2((x - min.x) / (max.x - min.x), (z - min.y) / (max.y - min.y));

    if (min.x < x && x < max.x && min.y < z && z < max.y && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, float2_as_ints(uv));
}

extern "C" __device__ void CH_FUNC(plane)()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    float3 local_n = make_float3(0, 1, 0);
    const float3 world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n));
    const float2 uv = getFloat2FromAttribute<0>();

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;
    si->shading.dpdu = optixTransformNormalFromObjectToWorldSpace(make_float3(1.0f, 0.0f, 0.0f));
    si->shading.dpdv = optixTransformNormalFromObjectToWorldSpace(make_float3(0.0f, 0.0f, 1.0f));
}