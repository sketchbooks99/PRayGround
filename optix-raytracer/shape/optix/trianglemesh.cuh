#pragma once

#include <sutil/vec_math.h>
#include "../../core/util.h"
#include "../../core/transform.h"
#include "../../core/material.h"
#include "../../optix/sbt.h"

namespace oprt {

struct MeshData {
    float3* vertices;
    int3* indices;
    float3* normals;
    float2* texcoords;
    // Transform transform;
};

}

#ifdef __CUDACC__

CALLABLE_FUNC void CH_FUNC(mesh)()
{
    oprt::HitGroupData* data = reinterpret_cast<oprt::HitGroupData*>(optixGetSbtDataPointer());
    const oprt::MeshData* mesh_data = reinterpret_cast<oprt::MeshData*>(data->shapedata);

    oprt::Ray ray = get_world_ray();
    
    const int prim_idx = optixGetPrimitiveIndex();
    const int3 index = mesh_data->indices[prim_idx];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float2 texcoord0 = mesh_data->texcoords[index.x];
    const float2 texcoord1 = mesh_data->texcoords[index.y];
    const float2 texcoord2 = mesh_data->texcoords[index.z];
    const float2 texcoords = (1-u-v)*texcoord0 + u*texcoord1 + v*texcoord2;

    float3 n0 = normalize(mesh_data->normals[index.x]);
	float3 n1 = normalize(mesh_data->normals[index.y]);
	float3 n2 = normalize(mesh_data->normals[index.z]);

    // Linear interpolation of normal by barycentric coordinates.
    float3 local_n = (1.0f-u-v)*n0 + u*n1 + v*n2;
    float3 world_n = optixTransformVectorFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    oprt::SurfaceInteraction* si = get_surfaceinteraction();
    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->wi = ray.d;
    si->uv = texcoords;

    si->mat_property = {
        data->matdata,              // matdata
        data->material_type * 2,    // bsdf_sample_idx
        data->material_type * 2 + 1 // pdf_idx
    };
}

// -------------------------------------------------------------------------------
CALLABLE_FUNC void CH_FUNC(mesh_occlusion)()
{
	setPayloadOcclusion(true);
}

#endif