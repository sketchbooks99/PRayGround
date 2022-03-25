#pragma once

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/core/material.h>
#include <prayground/core/ray.h>
#include <prayground/shape/plane.h>
#include <prayground/optix/cuda/util.cuh>

namespace prayground {

CALLABLE_FUNC void IS_FUNC(plane)()
{

    const HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const Plane::Data* plane_data = reinterpret_cast<Plane::Data*>(data->shape_data);

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

    si->surface_type = data->surface_type;
    si->surface_property = {
        data->surface_data,
        data->surface_func_base_id
    };
}

CALLABLE_FUNC void CH_FUNC(plane_occlusion)()
{
    setPayloadOcclusion(true);
}

}