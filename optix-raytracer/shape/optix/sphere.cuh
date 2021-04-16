#pragma once 

#include <sutil/vec_math.h>
#include "../../core/transform.h"
#include "../../core/material.h"
#include "../../optix/sbt.h"

namespace oprt {

struct SphereData {
    float3 center;
    float radius;
    // Transform transform;
};

}

#ifdef __CUDACC__

INLINE DEVICE float2 get_uv(float3 p) {
    float3 temp = p;
    float phi = atan2(temp.z, temp.x);
    float theta = asin(temp.y);
    float u = 1.0f - (phi + M_PIf) / (2.0f * M_PIf);
    float v = (theta + M_PIf/2.0f) / M_PIf;
    return make_float2(u, v);
}

CALLABLE_FUNC void IS_FUNC(sphere)() {
    const oprt::HitGroupData* data = reinterpret_cast<oprt::HitGroupData*>(optixGetSbtDataPointer());
    const oprt::SphereData* sphere_data = reinterpret_cast<oprt::SphereData*>(data->shapedata);

    const float3 center = sphere_data->center;
    const float radius = sphere_data->radius;

    oprt::Ray ray = get_local_ray();

    const float3 oc = ray.o - center;
    const float a = dot(ray.d, ray.d);
    const float half_b = dot(oc, ray.d);
    const float c = dot(oc, oc) - radius*radius;
    const float discriminant = half_b*half_b - a*c;

    if (discriminant > 0.0f) {
        float sqrtd = sqrtf(discriminant);
        float root1 = (-half_b - sqrtd) / a;
        bool check_second = true;
        if ( root1 > ray.tmin && root1 < ray.tmax ) {
            float3 normal = normalize((ray.o + normalize(ray.d) * root1 - center) / radius);
            optixReportIntersection(root1, 0, float3_as_ints(normal));
        }

        if (check_second) {
            float root2 = (-half_b + sqrtd) / a;
            if ( root2 > ray.tmin && root2 < ray.tmax ) {
                float3 normal = normalize((ray.o + normalize(ray.d) * root2 - center) / radius);
                optixReportIntersection(root2, 0, float3_as_ints(normal));
            }
        }
    }
}

CALLABLE_FUNC void CH_FUNC(sphere)() {
    const oprt::HitGroupData* data = reinterpret_cast<oprt::HitGroupData*>(optixGetSbtDataPointer());
    const oprt::SphereData* sphere_data = reinterpret_cast<oprt::SphereData*>(data->shapedata);

    oprt::Ray ray = get_world_ray();

    float3 local_n = make_float3(
        int_as_float( optixGetAttribute_0() ),
        int_as_float( optixGetAttribute_1() ),
        int_as_float( optixGetAttribute_2() )
    );
    float3 world_n = optixTransformVectorFromObjectToWorldSpace(local_n);
    world_n = normalize(faceforward(world_n, -ray.d, world_n));

    oprt::SurfaceInteraction* si = get_surfaceinteraction();
    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->wi = ray.d;
    si->uv = get_uv(local_n);

    // Sampling material properties.
    optixContinuationCall<void, oprt::SurfaceInteraction*, void*>(data->sample_func_idx, si, data->matdata);
}

#endif