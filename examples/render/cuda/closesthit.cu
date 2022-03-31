#include "util.cuh"
#include <prayground/optix/sbt.h>

extern "C" __device__ void __closesthit__shadow()
{
    setPayload<0>(1);
}

/// Custom primitives
/// @note Intersection programs are declared in prayground/optix/cuda/intersection.cu
extern "C" __device__ void __closesthit__custom()
{
    const auto* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    auto* shading = getPtrFromTwoAttributes<Shading, 0>();
    shading->n = optixTransformNormalFromObjectToWorldSpace(shading->n);
    shading->n = normalize(shading->n);
    shading->dpdu = optixTransformVectorFromObjectToWorldSpace(shading->dpdu);
    shading->dpdv = optixTransformVectorFromObjectToWorldSpace(shading->dpdv);

    auto* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    si->p = ray.at(ray.tmax);
    si->shading = *shading; 
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}

extern "C" __device__ void __closesthit__mesh()
{
    const auto* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const auto* mesh = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

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

    const Vec3f n0 = mesh->normals[face.normal_id.x()];
	const Vec3f n1 = mesh->normals[face.normal_id.y()];
	const Vec3f n2 = mesh->normals[face.normal_id.z()];

    // Linear interpolation of normal by barycentric coordinates.
    Shading shading;
    shading.uv = texcoords;

    Vec3f local_n = (1.0f-u-v)*n0 + u*n1 + v*n2;
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    shading.n = normalize(world_n);

    // Calculate partial derivative on texture coordinates
    const Vec2f duv02 = texcoord0 - texcoord2, duv12 = texcoord1 - texcoord2;
    const Vec3f dp02 = p0 - p2, dp12 = p1 - p2;
    const float D = duv02.x() * duv12.y() - duv02.y() * duv12.x();
    bool degenerateUV = abs(D) < 1e-8f;
    if (!degenerateUV)
    {
        const float invD = 1.0f / D;
        shading.dpdu = (duv12.y() * dp02 - duv02.y() * dp12) * invD;
        shading.dpdv = (-duv12.x() * dp02 + duv02.x() * dp12) * invD;
    }
    if (degenerateUV || length(cross(dpdu, dpdv)) == 0.0f)
    {
        const Vec3f n = normalize(cross(p2 - p0, p1 - p0));
        Onb onb(n);
        shading.dpdu = onb.tangent;
        shading.dpdv = onb.bitangent;
    }

    shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdu));
    shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdv));

    auto* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}