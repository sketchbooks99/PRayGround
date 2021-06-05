#pragma once

#include <sutil/vec_math.h>
#include "../../core/material.h"
#include "../../core/ray.h"
#include "../../optix/sbt.h"

namespace oprt {

struct PlaneData 
{
    float2 min;
    float2 max;
};

#ifdef __CUDACC__

CALLABLE_FUNC void IS_FUNC(plane)()
{

    const HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const PlaneData* plane_data = reinterpret_cast<PlaneData*>(data->shapedata);

    const float2 min = plane_data->min;
    const float2 max = plane_data->max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y / ray.d.y;

    const float x = ray.o.x + t * ray.d.x;
    const float z = ray.o.z + t * ray.d.z;

    float2 uv = make_float2(x / (max.x - min.x), z / (max.y - min.y));

    if (min.x < x && x < max.x && min.y < z && z < max.y && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, float2_as_ints(uv));
}

CALLABLE_FUNC void CH_FUNC(plane)()
{
    HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    float3 local_n = make_float3(0, 1, 0);
    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);
    float2 uv = make_float2(
        int_as_float( optixGetAttribute_0() ), 
        int_as_float( optixGetAttribute_1() )
    );

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->wi = ray.d;
    si->uv = uv;

    si->mat_property = {
        data->matdata,              // material data
        data->material_type * 2,    // bsdf_sample_id
        data->material_type * 2 + 1 // pdf_id
    };
}

CALLABLE_FUNC void CH_FUNC(plane_occlusion)()
{
    setPayloadOcclusion(true);
}

#endif

}