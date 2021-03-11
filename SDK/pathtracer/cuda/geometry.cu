#include <optix.h>

#include <cuda/random.h>
#include "../core/pathtracer.h"

#include <sutil/vec_math.h>
#include "helpers.h"

#include "cuda_util.h"

extern "C" {
__constant__ Params params;
}

/** MEMO: 
 * Write intersection algorithm, right now! */

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

    SurfaceInteaction* si = getSurfaceInteraction();

    if (discriminant > 0.0f) {
        float sqrtd = sqrtf(discriminant);
        bool near_valid = true, far_valid = true;

        float root = (-half_b - sqrtd) / a;
        near_valid = !(root < t_min || root > t_max); 
        root = (-half_b + sqrtd) / a;
        far_valid = !(root < t_min || root > t_max);

        if (near_valid && far_valid) {
            si->t = root;
            si->p = ray_orig*t + ray_dir;
            si->wi = normalize(ray_dir);
            vec3 normal = (si->p - hit_group_data->center) / radius;
            si->n = normal; 
            optixReportIntersection(t, 0, float3_as_ints(normal), float_as_int(radius));
        }
    }
}