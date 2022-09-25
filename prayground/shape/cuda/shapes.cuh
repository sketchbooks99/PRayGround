#pragma once 

#define PG_INTERSECTION(name) __intersection__pg_##name
#define PG_INTERSECTION_TEXT(name) "__intersection__pg_" name

#include <prayground/optix/cuda/device_util.cuh>
#include <prayground/shape/box.h>
#include <prayground/shape/cylinder.h>
#include <prayground/shape/plane.h>
#include <prayground/shape/sphere.h>
#include <prayground/core/ray.h>
#include <prayground/core/interaction.h>
#include <prayground/optix/sbt.h>

#ifdef __CUDACC__

namespace prayground {

    // ----------------------------------------------------------------------------------------
    // Cylinder
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec2f getCylinderUV(
        const Vec3f& p, const Cylinder::Data& cylinder, const bool hit_disk)
    {
        if (hit_disk)
        {
            const float r = sqrtf(p.x() * p.x() + p.z() * p.z()) / cylinder.radius;
            const float theta = atan2(p.z(), p.x());
            float u = 1.0f - (theta + math::pi / 2.0f) / math::pi;
            return Vec2f(u, r);
        }
        else
        {
            float phi = atan2(p.z(), p.x());
            if (phi < 0.0f) phi += math::two_pi;
            const float u = phi / math::two_pi;
            const float v = (p.y() + cylinder.height / 2.0f) / cylinder.height;
            return Vec2f(u, v);
        }
    }

    INLINE DEVICE bool intersectionCylinder(
        const Cylinder::Data* cylinder, const Ray& ray, Shading* shading, float* time
    )
    {
        const float radius = cylinder->radius;
        const float height = cylinder->height;

        // Get discriminant with infinite cylinder along with Y axis
        const float a = pow2(ray.d.x()) + pow2(ray.d.z());
        const float half_b = ray.o.x() * ray.d.x() + ray.o.z() * ray.d.z();
        const float c = dot(ray.o, ray.o) - ray.o.y() * ray.o.y() - radius * radius;
        const float discriminant = half_b * half_b - a * c;

        if (discriminant <= 0.0f)
            return false;

        const float sqrtd = sqrtf(discriminant);

        const float side_tmin = (-half_b - sqrtd) / a;
        const float side_tmax = (-half_b + sqrtd) / a;

        if (side_tmin > ray.tmax || side_tmax < ray.tmin)
            return false;

        const float upper = height / 2.0f;
        const float lower = -height / 2.0f;
        const float y_tmin = fmin((lower - ray.o.y()) / ray.d.y(), (upper - ray.o.y()) / ray.d.y());
        const float y_tmax = fmax((lower - ray.o.y()) / ray.d.y(), (upper - ray.o.y()) / ray.d.y());

        float tmin = fmax(y_tmin, side_tmin);
        float tmax = fmin(y_tmax, side_tmax);
        if (tmin > tmax || (tmax < ray.tmin) || (tmin > ray.tmax))
            return false;

        bool hit_min = true;
        float t = tmin;

        // Check near intersection
        if (t < ray.tmin || ray.tmax < t)
        {
            hit_min = false;
            t = tmax;
            // Check far intersection
            if (t < ray.tmin || ray.tmax < t)
                return false;
        }

        Vec3f p = ray.at(t);
        bool hit_disk = hit_min ? y_tmin > side_tmin : y_tmax < side_tmax;
        Vec3f n = hit_disk
            ? normalize(p - Vec3f(p.x(), 0.0f, p.z()))   // Hit at disk
            : normalize(p - Vec3f(0.0f, p.y(), 0.0f));   // Hit at side
        Vec2f uv = getCylinderUV(p, *cylinder, hit_disk);

        shading->n = n;
        shading->uv = uv;
        *time = t;

        if (hit_disk)
        {
            const float r_hit = sqrtf(p.x() * p.x() + p.z() * p.z());
            shading->dpdu = Vec3f(-math::two_pi * p.y(), 0.0f, math::two_pi * p.z());
            shading->dpdv = Vec3f(p.x(), 0.0f, p.z()) * radius / r_hit;
        }
        else
        {
            shading->dpdu = Vec3f(-math::two_pi * p.z(), 0.0f, math::two_pi * p.x());
            shading->dpdv = Vec3f(0.0f, height, 0.0f);
        }
        return true;
    }

    extern "C" DEVICE void PG_INTERSECTION(cylinder)()
    {
        const pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
        const Cylinder::Data* cylinder = reinterpret_cast<Cylinder::Data*>(data->shape_data);

        Ray ray = getLocalRay();

        Shading shading;
        float time;
        if (intersectionCylinder(cylinder, ray, &shading, &time))
        {
            // Pack shading pointer to two attributes
            // If you'd like to unpack the pointer, use
            // Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
            uint32_t a0, a1;
            packPointer(&shading, a0, a1);
            optixReportIntersection(time, 0, a0, a1);
        }
    }

    extern "C" DEVICE void PG_INTERSECTION(cylinder_instanced)()
    {
        const pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
        const int32_t idx = optixGetPrimitiveIndex();
        const Cylinder::Data cylinder = reinterpret_cast<Cylinder::Data*>(data->shape_data)[idx];

        Ray ray = getLocalRay();

        Shading shading;
        float time;
        if (intersectionCylinder(&cylinder, ray, &shading, &time))
        {
            // Pack shading pointer to two attributes
            // If you'd like to unpack the pointer, use
            // Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
            uint32_t a0, a1;
            packPointer(&shading, a0, a1);
            optixReportIntersection(time, 0, a0, a1);
        }
    }

    // ----------------------------------------------------------------------------------------
    // Box
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec2f getBoxUV(const Vec3f& p, const Box::Data& box, const int axis)
    {
        int u_axis = (axis + 1) % 3;
        int v_axis = (axis + 2) % 3;

        // Swap axis (u=Z, v=X) -> (u=X, v=Z) if axis == Y
        if (axis == 1) swap(u_axis, v_axis);

        Vec2f uv(
            (p[u_axis] - box.min[u_axis]) / (box.max[u_axis] - box.min[u_axis]),
            (p[v_axis] - box.min[v_axis]) / (box.max[v_axis] - box.min[v_axis])
        );

        return clamp(uv, 0.0f, 1.0f);
    }

    /* Return a hitting axis of box face, X=0, Y=1, Z=2 */
    INLINE DEVICE bool intersectionBox(
        const Box::Data* box, const Ray& ray, Shading* shading, float* time
    )
    {
        const Vec3f min = box->min;
        const Vec3f max = box->max;

        float tmin = ray.tmin, tmax = ray.tmax;
        int min_axis = -1, max_axis = -1;

        // Intersection test
        for (int i = 0; i < 3; i++)
        {
            float t0, t1;
            if (ray.d[i] == 0.0f)
            {
                t0 = fminf(min[i] - ray.o[i], max[i] - ray.o[i]);
                t1 = fmaxf(min[i] - ray.o[i], max[i] - ray.o[i]);
            }
            else
            {
                t0 = fminf((min[i] - ray.o[i]) / ray.d[i], (max[i] - ray.o[i]) / ray.d[i]);
                t1 = fmaxf((min[i] - ray.o[i]) / ray.d[i], (max[i] - ray.o[i]) / ray.d[i]);
            }

            // Update hitting axis 
            min_axis = t0 > tmin ? i : min_axis;
            max_axis = t1 < tmax ? i : max_axis;

            // Update distance from the ray origin to an intersection surface
            tmin = fmaxf(t0, tmin);
            tmax = fminf(t1, tmax);

            // No intersection
            if (tmax < tmin)
                return false;
        }

        Vec3f center = (min + max) / 2.0f;
        int axis = min_axis;
        float t = tmin;
        // Check if the smaller ray time is valid
        if ((t < ray.tmin || ray.tmax < t) || (axis < 0 || 2 < axis))
        {
            axis = max_axis;
            t = tmax;
            // Check if the bigger ray time is valid
            if ((t < ray.tmin || ray.tmax < t) || (axis < 0 || 2 < axis))
                return false;
        }

        Vec3f p = ray.at(t);
        Vec3f center_axis = p;
        center_axis[axis] = center[axis];
        Vec3f n = normalize(p - center_axis);
        Vec2f uv = getBoxUV(p, *box, axis);

        // Store the shading information
        shading->n = n;
        shading->uv = uv;

        // x
        if (axis == 0)
        {
            shading->dpdu = Vec3f(0.0f, 0.0f, 1.0f);
            shading->dpdv = Vec3f(0.0f, 1.0f, 0.0f);
        }
        // y
        else if (axis == 1)
        {
            shading->dpdu = Vec3f(1.0f, 0.0f, 0.0f);
            shading->dpdv = Vec3f(0.0f, 0.0f, 1.0f);
        }
        // z
        else if (axis == 2)
        {
            shading->dpdu = Vec3f(1.0f, 0.0f, 0.0f);
            shading->dpdv = Vec3f(0.0f, 1.0f, 0.0f);
        }

        *time = t;

        return true;
    }

    extern "C" void DEVICE PG_INTERSECTION(box)()
    {
        const pgHitgroupData* hg_data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
        const Box::Data* box = reinterpret_cast<Box::Data*>(hg_data->shape_data);

        Ray ray = getLocalRay();

        Shading shading;
        float time;
        if (intersectionBox(box, ray, &shading, &time))
        {
            // Pack shading pointer to two attributes
            // If you'd like to unpack the pointer, use
            // Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
            uint32_t a0, a1;
            packPointer(&shading, a0, a1);
            optixReportIntersection(time, 0, a0, a1);
        }
    }

    extern "C" void DEVICE PG_INTERSECTION(box_instanced)()
    {
        const pgHitgroupData* hg_data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
        const int32_t idx = optixGetPrimitiveIndex();
        const Box::Data box = reinterpret_cast<Box::Data*>(hg_data->shape_data)[idx];

        Ray ray = getLocalRay();

        Shading shading;
        float time;
        if (intersectionBox(&box, ray, &shading, &time))
        {
            // Pack shading pointer to two attributes
            // If you'd like to unpack the pointer, use
            // Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
            uint32_t a0, a1;
            packPointer(&shading, a0, a1);
            optixReportIntersection(time, 0, a0, a1);
        }
    }

    // ----------------------------------------------------------------------------------------
    // Plane
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec2f getPlaneUV(const Vec2f p, const Plane::Data& plane)
    {
        const float u = (p.x() - plane.min.x()) / (plane.max.x() - plane.min.x());
        const float v = (p.y() - plane.min.y()) / (plane.max.y() - plane.min.y());
        return Vec2f(u, v);
    }

    INLINE DEVICE bool intersectionPlane(const Plane::Data* plane, const Ray& ray, Shading* shading, float* time)
    {
        const float t = -ray.o.y() / ray.d.y();
        const float x = ray.o.x() + t * ray.d.x();
        const float z = ray.o.z() + t * ray.d.z();

        if (plane->min.x() < x && x < plane->max.x() &&
            plane->min.y() < z && z < plane->max.y() &&
            ray.tmin < t && t < ray.tmax)
        {
            shading->uv = getPlaneUV(Vec2f(x, z), *plane);
            shading->n = Vec3f(0, 1, 0);
            shading->dpdu = Vec3f(1, 0, 0);
            shading->dpdv = Vec3f(0, 0, 1);
            *time = t;
            return true;
        }
        return false;
    }

    extern "C" DEVICE void PG_INTERSECTION(plane)()
    {
        const pgHitgroupData* hg_data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
        const Plane::Data* plane = reinterpret_cast<Plane::Data*>(hg_data->shape_data);

        Ray ray = getLocalRay();

        Shading shading;
        float time;
        if (intersectionPlane(plane, ray, &shading, &time))
        {
            // Pack shading pointer to two attributes
            // If you'd like to unpack the pointer, use
            // Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
            uint32_t a0, a1;
            packPointer(&shading, a0, a1);
            optixReportIntersection(time, 0, a0, a1);
        }
    }

    extern "C" DEVICE void PG_INTERSECTION(plane_instanced)()
    {
        const pgHitgroupData* hg_data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
        const int32_t idx = optixGetPrimitiveIndex();
        const Plane::Data plane = reinterpret_cast<Plane::Data*>(hg_data->shape_data)[idx];

        Ray ray = getLocalRay();

        Shading shading;
        float time;
        if (intersectionPlane(&plane, ray, &shading, &time))
        {
            // Pack shading pointer to two attributes
            // If you'd like to unpack the pointer, use
            // Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
            uint32_t a0, a1;
            packPointer(&shading, a0, a1);
            optixReportIntersection(time, 0, a0, a1);
        }
    }

    // ----------------------------------------------------------------------------------------
    // Sphere
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec2f getSphereUV(const Vec3f& p) {
        float phi = atan2(p.z(), p.x());
        if (phi < 0) phi += 2.0f * math::pi;
        float theta = acos(p.y());
        float u = phi / (2.0f * math::pi);
        float v = theta * math::inv_pi;
        return Vec2f(u, v);
    }

    INLINE DEVICE bool intersectionSphere(
        const Sphere::Data* sphere, const Ray& ray, Shading* shading, float* time)
    {
        const Vec3f oc = ray.o - sphere->center;
        const float a = dot(ray.d, ray.d);
        const float half_b = dot(oc, ray.d);
        const float c = dot(oc, oc) - pow2(sphere->radius);
        const float discriminant = half_b * half_b - a * c;

        if (discriminant <= 0.0f)
            return false;

        const float sqrtd = sqrtf(discriminant);

        float t = (-half_b - sqrtd) / a;
        if (t < ray.tmin || ray.tmax < t)
        {
            t = (-half_b + sqrtd) / a;
            if (t < ray.tmin || ray.tmax < t)
                return false;
        }

        const Vec3f p = ray.at(t);
        shading->n = p / sphere->radius;
        shading->uv = getSphereUV(shading->n);

        float phi = atan2(shading->n.z(), shading->n.x());
        if (phi < 0) phi += math::two_pi;
        const float theta = acosf(shading->n.y());
        shading->dpdu = Vec3f(-math::two_pi * shading->n.z(), 0, math::two_pi * shading->n.x());
        shading->dpdv = math::pi * Vec3f(shading->n.y() * cosf(phi), -sinf(theta), shading->n.y() * sinf(phi));

        *time = t;

        return true;
    }

    extern "C" DEVICE void PG_INTERSECTION(sphere)()
    {
        const pgHitgroupData* hg_data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
        const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(hg_data->shape_data);

        Ray ray = getLocalRay();

        Shading shading;
        float time;
        if (intersectionSphere(sphere, ray, &shading, &time))
        {
            // Pack shading pointer to two attributes
            // If you'd like to unpack the pointer, use
            // Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
            uint32_t a0, a1;
            packPointer(&shading, a0, a1);
            optixReportIntersection(time, 0, a0, a1);
        }
    }

    extern "C" DEVICE void PG_INTERSECTION(sphere_instanced)()
    {
        const pgHitgroupData* hg_data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
        const int32_t idx = optixGetPrimitiveIndex();
        const Sphere::Data sphere = reinterpret_cast<Sphere::Data*>(hg_data->shape_data)[idx];

        Ray ray = getLocalRay();

        Shading shading;
        float time;
        if (intersectionSphere(&sphere, ray, &shading, &time))
        {
            // Pack shading pointer to two attributes
            // If you'd like to unpack the pointer, use
            // Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
            uint32_t a0, a1;
            packPointer(&shading, a0, a1);
            optixReportIntersection(time, 0, a0, a1);
        }
    }

    // ----------------------------------------------------------------------------------------
    // Curves
    // ----------------------------------------------------------------------------------------
    
    // Return shading frame contains the NORMAL, TEXCOORD, derivatives on point/normal (dpdu, dpdv, dndu, dndv)
    INLINE DEVICE Shading getShadingCurvesLinear(const uint32_t primitive_idx)
    {
        const OptixTraversableHandle gas = optixGetGASTraversableHandle();
        const uint32_t gas_sbt_idx = optixGetSbtGASIndex();
        float4 control_points[2];

        optixGetLinearCurveVertexData(gas, primitive_idx, gas_sbt_idx, 0.0f, control_points);

        LinearInterpolator interpolator;
        interpolator.initialize(control_points);

        const float u = optixGetCurveParameter;

        Vec4f v0 = control_points[0], v1 = control_points[1];
        Vec4f velocity = v1 - v0;

        Ray ray = getLocalRay();
        Vec3f hit_point = ray.at(ray.tmax);

        // Curve point and width on the intersection detected
        Vec3f p = Vec3f(v0) + u * Vec3f(velocity);
        float w = v0.w() + u * velocity.w();

        float dd = dot(Vec3f(velocity), Vec3f(velocity));

        Vec3f o1 = hit_point - p;
        o1 -= (dot(o1, Vec3f(velocity)) / dd) * Vec3f(velocity);
        o1 *= w / length(o1);
        hit_point = p + o1;

        Ray ray = getLocalRay();
        Vec3f hit_point = ray.at(ray.tmax);
        Vec3f n = dd * o1 - (velocity.w() * w) * Vec3f(velocity);

        Shading shading;
        shading.u = Vec2f(0.0f, u);
        shading.n = n;
        shading.dpdv = normalize(Vec3f(velocity));
        shading.dpdu = cross(shading.n, shading.dpdv);
    }

    INLINE DEVICE Shading getShadingCurves(const uint32_t primitive_idx, OptixPrimitiveType primitive_type)
    {
        switch (primitive_type)
        {
        case OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE:
        case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
            return getShadingCurvesLinear(primitive_idx);
        case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE:
        case OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM:
        }
    }

} // namespace prayground

#endif // __CUDACC__
