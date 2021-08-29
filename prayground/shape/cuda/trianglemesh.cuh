#pragma once

#include <prayground/math/vec_math.h>
#include <prayground/optix/cuda/util.cuh>
#include <prayground/shape/trianglemesh.h>

namespace prayground {

CALLABLE_FUNC void CH_FUNC(mesh)()
{
    HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const MeshData* mesh_data = reinterpret_cast<MeshData*>(data->shape_data);

    Ray ray = getWorldRay();
    
    const int prim_id = optixGetPrimitiveIndex();
    const Face face = mesh_data->faces[prim_id];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float2 texcoord0 = mesh_data->texcoords[face.texcoord_id.x];
    const float2 texcoord1 = mesh_data->texcoords[face.texcoord_id.y];
    const float2 texcoord2 = mesh_data->texcoords[face.texcoord_id.z];
    const float2 texcoords = (1-u-v)*texcoord0 + u*texcoord1 + v*texcoord2;

    float3 n0 = normalize(mesh_data->normals[face.normal_id.x]);
	float3 n1 = normalize(mesh_data->normals[face.normal_id.y]);
	float3 n2 = normalize(mesh_data->normals[face.normal_id.z]);

    // Linear interpolation of normal by barycentric coordinates.
    float3 local_n = (1.0f-u-v)*n0 + u*n1 + v*n2;
    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->wi = ray.d;
    si->uv = texcoords;

    si->surface_type = data->surface_type;
    si->surface_property = {
        data->surface_data,
        data->surface_func_base_id
    };
}

// -------------------------------------------------------------------------------
CALLABLE_FUNC void CH_FUNC(mesh_occlusion)()
{
	setPayloadOcclusion(true);
}

