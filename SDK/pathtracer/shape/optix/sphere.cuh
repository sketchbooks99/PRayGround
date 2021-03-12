#pragma once 

#include <core/util.h>
#include <sutil/vec_math.h>
#include <core/transform.h>
#include <core/material.h>

namespace pt {

struct SphereHitGroupData {
    float3 center;
    float radius;
    Transform transform;
    MaterialPtr matptr;
};

INLINE DEVICE float2 getUV()

CALLABLE_FUNC void IS_FUNC(sphere) {
    const pt::SphereHitGroupData* sphere_data = reinterpret_cast<pt::SphereHitGroupData*>(optixGetSbtDataPointer());

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    const float3 oc = ray_orig - sphere_data->center;
    const float a = dot(ray_dir, ray_dir);
    const float hal_b = dot(oc, ray_dir);
    const float c = dot(oc, oc) - radius*radius;
    const float discriminant = half_b*half_b - a*c;

    if (discriminant > 0.0f) {
        float sqrtd = sqrtf(discriminant);
        bool near_valid = true, far_valid = true;

        float root = (-half_b - sqrtd) / a;
        near_valid = !(root < t_min || root > t_max); 
        root = (-half_b + sqrtd) / a;
        far_valid = !(root < t_min || root > t_max);

        if (near_valid && far_valid) {
            vec3 normal = (si->p - hit_group_data->center) / radius;
            optixReportIntersection(t, 0, float3_as_ints(normal));
        }
    }
}

CALLABLE_FUNC void CH_FUNC(sphere) {
    const SphereHitGroupData* sphere_data = reinterpret_cast<SphereHitGroupData*>(optixGetSbtDataPointer());

    const float3 ro = optixGetWorldRayOrigin();
    const float3 rd = optixGetWorldRayDirection();
    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();

    float3 n = make_float3(
        int_as_float( optixGetAttribute_0() ),
        int_as_float( optixGetAttribute_1() ),
        int_as_float( optixGetAttribute_2() )
    );

    n = faceforward(n, -rd, n);

    SurfaceInteaction* si = get_surfaceinteraction();
    si.p = ro + tmax*rd;
    si.n = n;
    si.wi = rd;
    sphere_data->matptr->sample(*si);
})

}