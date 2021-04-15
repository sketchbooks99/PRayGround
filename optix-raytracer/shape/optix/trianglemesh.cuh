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
    
    const int prim_idx = optixGetPrimitiveIndex();
    const int3 index = mesh_data->indices[prim_idx];
    const float3 ro = optixGetWorldRayOrigin();
    const float3 rd = optixGetWorldRayDirection();
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float2 texcoord0 = mesh_data->texcoords[index.x];
    const float2 texcoord1 = mesh_data->texcoords[index.y];
    const float2 texcoord2 = mesh_data->texcoords[index.z];
    const float2 texcoords = (1-u-v)*texcoord0 + u*texcoord1 + v*texcoord2;

    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();

	float3 n0 = normalize(mesh_data->normals[index.x]);
	float3 n1 = normalize(mesh_data->normals[index.y]);
	float3 n2 = normalize(mesh_data->normals[index.z]);
    // n0 = optixTransformVectorFromWorldToObjectSpace(n0);
    // n1 = optixTransformVectorFromWorldToObjectSpace(n1);
    // n2 = optixTransformVectorFromWorldToObjectSpace(n2);
    n0 = optixTransformVectorFromObjectToWorldSpace(n0);
    n1 = optixTransformVectorFromObjectToWorldSpace(n1);
    n2 = optixTransformVectorFromObjectToWorldSpace(n2);

    // Linear interpolation of normal by barycentric coordinates.
    float3 n = normalize( (1.0f-u-v)*n0 + u*n1 + v*n2 );
    n = faceforward(n, -rd, n);

    oprt::SurfaceInteraction* si = get_surfaceinteraction();
    si->p = ro + tmax*rd;
    si->n = n;
    si->wi = rd;
    si->uv = texcoords;

    // Sampling material properties.
    optixContinuationCall<void, oprt::SurfaceInteraction*, void*>(data->sample_func_idx, si, data->matdata);
    // si->attenuation = make_float3(texcoords.x, texcoords.y, 0.5f);
}

#endif