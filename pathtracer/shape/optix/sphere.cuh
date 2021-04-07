#pragma once 

#include <sutil/vec_math.h>
#include <include/core/transform.h>
#include <include/core/material.h>
#include <include/optix/sbt.h>

namespace pt {

struct SphereData {
    float3 center;
    float radius;
    // Transform transform;
};

}

#ifdef __CUDACC__

CALLABLE_FUNC void IS_FUNC(sphere)() {
    const pt::HitGroupData* data = reinterpret_cast<pt::HitGroupData*>(optixGetSbtDataPointer());
    const pt::SphereData* sphere_data = reinterpret_cast<pt::SphereData*>(data->shapedata);

    const float3 center = sphere_data->center;
    const float radius = sphere_data->radius;

    optixReportIntersection(0.f, 0, float3_as_ints(make_float3(1, 0, 0)));

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float tmin = optixGetRayTmin();
    const float tmax = optixGetRayTmax();

    const float3 oc = ray_orig - center;
    const float a = dot(ray_dir, ray_dir);
    const float half_b = dot(oc, ray_dir);
    const float c = dot(oc, oc) - radius*radius;
    const float discriminant = half_b*half_b - a*c;

    if (discriminant > 0.0f) {
        float sqrtd = sqrtf(discriminant);
        float root1 = (-half_b - sqrtd) / a;
        bool check_second = true;
        if ( root1 > tmin && root1 < tmax ) {
            float3 normal = (ray_orig + normalize(ray_dir) * root1 - center) / radius;
            optixReportIntersection(root1, 0, float3_as_ints(normal));
        }

        if (check_second) {
            float root2 = (-half_b + sqrtd) / a;
            if ( root2 > tmin && root2 < tmax ) {
                float3 normal = (ray_orig + normalize(ray_dir) * root2 - center) / radius;
                optixReportIntersection(root2, 0, float3_as_ints(normal));
            }
        }
    }
}

CALLABLE_FUNC void CH_FUNC(sphere)() {
    const pt::HitGroupData* data = reinterpret_cast<pt::HitGroupData*>(optixGetSbtDataPointer());
    const pt::SphereData* sphere_data = reinterpret_cast<pt::SphereData*>(data->shapedata);
    const pt::Material* matptr = data->matptr;

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

    pt::SurfaceInteraction* si = get_surfaceinteraction();
    si->p = ro + tmax*rd;
    si->n = n;
    si->wi = rd;
    /** 
     * \note This member function wkll causes the error of illegal memory access 
     * or invalid program counter errordue to a wrong allocation of material pointer
     * on the device.
     */
    // matptr->sample(*si);
    si->radiance = make_float3(n.x, n.y, 0.5f);
}

#endif