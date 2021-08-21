#pragma once 

#include <sutil/vec_math.h>
#include <oprt/core/ray.h>
#include <oprt/core/interaction.h>
#include <oprt/shape/sphere.h>
#include <oprt/optix/cuda/util.cuh>

namespace oprt {

static INLINE DEVICE float2 getUV(const float3& p) {
    float phi = atan2(p.z, p.x);
    float theta = asin(p.y);
    float u = 1.0f - (phi + M_PIf) / (2.0f * M_PIf);
    float v = 1.0f - (theta + M_PIf/2.0f) / M_PIf;
    return make_float2(u, v);
}

CALLABLE_FUNC void IS_FUNC(sphere)() {
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const SphereData* sphere_data = reinterpret_cast<SphereData*>(data->shape_data);

    const float3 center = sphere_data->center;
    const float radius = sphere_data->radius;

    Ray ray = getLocalRay();

    const float3 oc = ray.o - center;
    const float a = dot(ray.d, ray.d);
    const float half_b = dot(oc, ray.d);
    const float c = dot(oc, oc) - radius*radius;
    const float discriminant = half_b*half_b - a*c;

    if (discriminant > 0.0f) {
        float sqrtd = sqrtf(discriminant);
        float t1 = (-half_b - sqrtd) / a;
        bool check_second = true;
        if ( t1 > ray.tmin && t1 < ray.tmax ) {
            float3 normal = normalize((ray.at(t1) - center) / radius);
            check_second = false;
            optixReportIntersection(t1, 0, float3_as_ints(normal));
        }

        if (check_second) {
            float t2 = (-half_b + sqrtd) / a;
            if ( t2 > ray.tmin && t2 < ray.tmax ) {
                float3 normal = normalize((ray.at(t2) - center) / radius);
                optixReportIntersection(t2, 0, float3_as_ints(normal));
            }
        }
    }
}

CALLABLE_FUNC void CH_FUNC(sphere)() {
    const HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const SphereData* sphere_data = reinterpret_cast<SphereData*>(data->shape_data);

    Ray ray = getWorldRay();

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
    si->uv = getUV(local_n);

    si->surface_type = data->surface_type;
    si->surface_property = {
        data->surface_data,
        data->surface_func_base_id
    };
}

// -------------------------------------------------------------------------------
CALLABLE_FUNC void CH_FUNC(sphere_occlusion)()
{
	setPayloadOcclusion(true);
}

}