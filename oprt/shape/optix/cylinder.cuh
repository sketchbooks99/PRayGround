#pragma once

#include <sutil/vec_math.h>
#include "../../core/material.h"
#include "../../core/ray.h"
#include "../../optix/sbt.h"

namespace oprt {

struct CylinderData
{
    float radius; 
    float height;
};

#ifdef __CUDACC__

CALLABLE_FUNC void IS_FUNC(cylinder)()
{
    const HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const CylinderData* cylinder = reinterpret_cast<CylinderData*>(data->shapedata);

    const float radius = cylinder->radius;
    const float height = cylinder->height;

    Ray ray = getLocalRay();
    
    const float a = dot(ray.d, ray.d) - ray.d.y * ray.d.y;
    const float half_b = (ray.o.x * ray.d.x + ray.o.z * ray.d.z);
    const float c = dot(ray.o, ray.o) - ray.o.y * ray.o.y - radius*radius;
    const float discriminant = half_b*half_b - a*c;

    /**
     * @todo
     * Implement test for upper and lower of cylinder.
     */

    if (discriminant > 0.0f)
    {
        float sqrtd = sqrtf(discriminant);
        float t1 = (-half_b - sqrtd) / a;
        bool check_second = true;
        if ( ray.tmin < t1 && t1 < ray.tmax )
        {
            float3 P = ray.at(t1);
            float3 normal = normalize((P - make_float3(P.x, 0.0f, P.z)) / radius);
            check_second = false;
            optixReportIntersection(t1, 0, float3_as_ints(normal));
        }

        if (check_second)
        {
            float t2 = (-half_b + sqrtd) / a;
            if ( ray.tmin < t2 && t2 < ray.tmax )
            {
                float3 P = ray.at(t2);
                float3 normal = normalize((P - make_float3(P.x, 0.0f, P.z)) / radius);
                optixReportIntersection(t2, 0, float3_as_ints(normal));
            }
        }
    }
}

CALLABLE_FUNC void CH_FUNC(cylinder)()
{
    const HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const CylinderData* cylinder = reinterpret_cast<CylinderData*>(data->shapedata);

    oprt::Ray ray = getWorldRay();

    float3 local_n = make_float3(
        int_as_float( optixGetAttribute_0() ),
        int_as_float( optixGetAttribute_1() ), 
        int_as_float( optixGetAttribute_2() )
    );
    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->wi = ray.d;
    si->uv = make_float2(1.0f);

    si->mat_property = {
        data->matdata,               // material data
        data->material_type * 2,     // id of callable function to evaluate bsdf and for importance sampling
        data->material_type * 2 + 1  // id of callable function to evaluate pdf
    };
}

CALLABLE_FUNC void CH_FUNC(cylinder_occlusion)()
{
    setPayloadOcclusion(true);
}

#endif

}