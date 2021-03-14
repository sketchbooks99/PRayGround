#pragma once

#include <core/util.h>
#include <sutil/vec_math.h>
#include <core/transform.h>
#include <core/material.h>

namespace pt {

struct MeshHitGroupData {
    float3* vertices;
    float3* normals;
    int3* indices;
    Transform transform;
    MaterialPtr matptr;
};

CALLABLE_FUNC void CH_FUNC(mesh)()
{
    const MeshHitGroupData* mesh_data = reinterpret_cast<MeshHitGroupData*>(optixGetSbtDataPointer());

    const int prim_idx = optixGetPrimitiveIndex();
    const int3 index = mesh_data->indices[prim_idx];
    const float3 ro = optixGetWorldRayOrigin();
    const float3 rd = optixGetWorldRayDirection();
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();

	const float3 n0 = normalize(mesh_data->normals[index.x]);
	const float3 n1 = normalize(mesh_data->normals[index.y]);
	const float3 n2 = normalize(mesh_data->normals[index.z]);

    // Linear interpolation of normal by barycentric coordinates.
    float3 n = normalize( (1.0f-u-v)*n0 + u*n1 + v*n2 );
    n = faceforward(n, -rd, n);

    SurfaceInteraction* si = get_surfaceinteraction();
    si->p = ro + tmax*rd;
    si->n = n;
    si->wi = rd;
    mesh_data->matptr->sample(*si);
}

}